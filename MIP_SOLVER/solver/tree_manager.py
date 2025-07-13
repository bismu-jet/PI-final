import time
import yaml
import math
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional, Tuple, Any

from solver.cuts import generate_all_cuts
from solver.problem import MIPProblem
from solver.node import Node
from solver.gurobi_interface import solve_lp_relaxation
from solver.heuristics import find_initial_solution
from solver.utilities import setup_logger
from solver.presolve import presolve

logger = setup_logger()

class TreeManager:
    """
    Manages the Branch and Bound tree, implementing the core solver logic.
    """
    def __init__(self, problem_path: str, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.problem = MIPProblem(problem_path)
        
        self.is_maximization = self.problem.model.ModelSense == gp.GRB.MAXIMIZE
        model_sense = "MAXIMIZE" if self.is_maximization else "MINIMIZE"
        logger.info(f"Problem recognized as a {model_sense} problem.")
        
        logger.info("--- Starting Presolve Phase ---")
        presolve(self.problem)
        logger.info("--- Presolve Phase Finished ---")
        
        self.active_nodes: List[Node] = []
        self.incumbent_solution: Optional[Dict[str, float]] = None
        self.incumbent_objective: Optional[float] = None
        
        self.global_best_bound: float = -math.inf if self.is_maximization else math.inf
        
        self.node_counter: int = 0
        self.optimality_gap = self.config['solver_params']['optimality_gap']
        self.time_limit_seconds = self.config['solver_params']['time_limit_seconds']

        self.cut_pool: List[Dict[str, Any]] = []
        
        self.switched_to_best_bound = False

        self.pseudocosts = {}

        logger.info(f"Initialized TreeManager with problem: {problem_path} and config: {config_path}")

    def _is_integer_feasible(self, solution: Dict[str, float], tolerance: float = 1e-6) -> bool:
        for var_name in self.problem.integer_variable_names:
            if var_name in solution:
                if abs(solution[var_name] - round(solution[var_name])) > tolerance:
                    return False
        return True

    def _update_pseudocosts(self, var_name: str, direction: str, degradation: float):
        if var_name not in self.pseudocosts:
            self.pseudocosts[var_name] = {
                'up': {'sum_degrad': 0.0, 'count': 0},
                'down': {'sum_degrad': 0.0, 'count': 0}
            }
        
        self.pseudocosts[var_name][direction]['sum_degrad'] += degradation
        self.pseudocosts[var_name][direction]['count'] += 1
        logger.debug(f"Updated pseudocost for '{var_name}' ({direction}): degradation={degradation:.4f}")

    def _select_by_pseudocost(self, solution: Dict[str, float], fractional_vars: List[str]) -> str:
        best_var = None
        max_score = -1.0
        
        for var_name in fractional_vars:
            val = solution[var_name]
            frac_part = val - math.floor(val)
            
            pc_down_info = self.pseudocosts.get(var_name, {}).get('down', {})
            pc_up_info = self.pseudocosts.get(var_name, {}).get('up', {})

            # --- FIX: Use .get() with a default value to prevent KeyError ---
            pc_down_count = pc_down_info.get('count', 0)
            pc_up_count = pc_up_info.get('count', 0)

            pc_down = (pc_down_info.get('sum_degrad', 0.0) / pc_down_count) if pc_down_count > 0 else 1.0
            pc_up = (pc_up_info.get('sum_degrad', 0.0) / pc_up_count) if pc_up_count > 0 else 1.0
            # --- END FIX ---
            
            score = (1 - frac_part) * pc_down + frac_part * pc_up
            
            if score > max_score:
                max_score = score
                best_var = var_name
                
        logger.info(f"Pseudocost choice: '{best_var}' with score {max_score:.4f}")
        return best_var

    def _get_branching_variable(self, solution: Dict[str, float], current_constraints: List[Tuple[str, str, float]]) -> Optional[str]:
        strategy = self.config["strategy"]["branching_variable"]

        fractional_vars = [
            var_name for var_name in self.problem.integer_variable_names
            if var_name in solution and abs(solution[var_name] - round(solution[var_name])) > 1e-6
        ]

        if not fractional_vars:
            return None

        if strategy == "most_fractional":
            return max(fractional_vars, key=lambda v: 0.5 - abs(solution[v] - math.floor(solution[v]) - 0.5))
        
        elif strategy == "pseudocost":
            return self._select_by_pseudocost(solution, fractional_vars)

        else:
            logger.warning(f"Unsupported branching variable strategy: {strategy}, falling back to most_fractional.")
            return max(fractional_vars, key=lambda v: 0.5 - abs(solution[v] - math.floor(solution[v]) - 0.5))

    def _find_violated_pool_cuts(self, solution: Dict[str, float]) -> List[Dict[str, Any]]:
        violated_cuts = []
        for cut in self.cut_pool:
            lhs_val = sum(coeff * solution.get(var_name, 0) for var_name, coeff in cut['coeffs'].items())
            rhs = cut['rhs']
            sense = cut['sense']
            
            is_violated = False
            if sense == GRB.LESS_EQUAL and lhs_val > rhs + 1e-6:
                is_violated = True
            elif sense == GRB.GREATER_EQUAL and lhs_val < rhs - 1e-6:
                is_violated = True
            
            if is_violated:
                violated_cuts.append(cut)
        
        if violated_cuts:
            logger.info(f"Found {len(violated_cuts)} violated cuts in the pool.")
        return violated_cuts

    def solve(self):
        logger.info("Starting Branch and Bound solver...")

        root_node = Node(node_id=self.node_counter, parent_id=None, local_constraints=[], lp_objective=None, lp_solution=None, status='PENDING')
        self.node_counter += 1

        logger.info(f"Solving root node {root_node.node_id} LP relaxation...")
        lp_result = solve_lp_relaxation(self.problem, root_node.local_constraints)

        if lp_result['status'] == 'OPTIMAL':
            root_node.lp_objective = lp_result['objective']
            root_node.lp_solution = lp_result['solution']
            root_node.vbasis = lp_result.get('vbasis')
            root_node.cbasis = lp_result.get('cbasis')
            root_node.status = 'SOLVED'
            self.global_best_bound = root_node.lp_objective
            self.active_nodes.append(root_node)
            logger.info(f"Root node {root_node.node_id} solved. LP Objective: {root_node.lp_objective:.4f}")

            candidate_integer_solution = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints)
            if candidate_integer_solution:
                fixed_vars_constraints = [(v, '==', float(round(val))) for v, val in candidate_integer_solution.items() if v in self.problem.integer_variable_names]
                completion_lp_result = solve_lp_relaxation(self.problem, fixed_vars_constraints)
                if completion_lp_result['status'] == 'OPTIMAL':
                    self.incumbent_solution = completion_lp_result['solution']
                    self.incumbent_objective = completion_lp_result['objective']
                    logger.info(f"Found and verified initial incumbent solution via heuristic. Objective: {self.incumbent_objective:.4f}")
                else:
                    logger.warning("Heuristic solution was not extendable to a feasible solution.")
        else:
            logger.error(f"Root node LP failed with status: {lp_result['status']}. Terminating.")
            return None, None

        start_time = time.time()
        while self.active_nodes:
            if time.time() - start_time > self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break
            
            if self.is_maximization:
                self.global_best_bound = max((node.lp_objective for node in self.active_nodes if node.lp_objective is not None), default=-math.inf)
            else:
                self.global_best_bound = min((node.lp_objective for node in self.active_nodes if node.lp_objective is not None), default=math.inf)
            
            logger.info(f"--- Nodes: {len(self.active_nodes)}, Global Best Bound: {self.global_best_bound:.4f}, Incumbent: {self.incumbent_objective} ---")

            if self.incumbent_objective is not None:
                if abs(self.incumbent_objective) > 1e-9:
                    gap = abs(self.incumbent_objective - self.global_best_bound) / abs(self.incumbent_objective)
                    if gap <= self.optimality_gap:
                        logger.info(f"Optimality gap ({gap:.6f}) reached {self.optimality_gap}. Terminating solver.")
                        break

            node_selection_strategy = self.config['strategy']['node_selection']
            if node_selection_strategy == 'best_bound':
                current_node = min(self.active_nodes, key=lambda node: node.lp_objective) if not self.is_maximization else max(self.active_nodes, key=lambda node: node.lp_objective)
                self.active_nodes.remove(current_node)
            else: # 'depth_first' or 'hybrid' start
                current_node = self.active_nodes.pop()

            logger.info(f"Processing node {current_node.node_id}. LP Objective: {current_node.lp_objective:.4f}")
            
            # --- FIX: Restored the cut generation and application logic ---
            if not self._is_integer_feasible(current_node.lp_solution):
                cuts_to_add = self._find_violated_pool_cuts(current_node.lp_solution)
                
                lp_result_for_cuts = {
                    'solution': current_node.lp_solution,
                    'vbasis': current_node.vbasis,
                    'cbasis': current_node.cbasis,
                    'local_constraints': current_node.local_constraints 
                }
                new_cuts = generate_all_cuts(self.problem, lp_result_for_cuts)
                
                for cut in new_cuts:
                    if cut not in self.cut_pool:
                        self.cut_pool.append(cut)
                
                cuts_to_add.extend(new_cuts)
                
                if cuts_to_add:
                    logger.info(f"Found/Generated {len(cuts_to_add)} cuts. Re-solving LP for node {current_node.node_id}.")
                    lp_result_after_cuts = solve_lp_relaxation(self.problem, current_node.local_constraints, cuts=cuts_to_add)
                    
                    if lp_result_after_cuts['status'] == 'OPTIMAL':
                         logger.info(f"LP re-solve successful. New LP objective: {lp_result_after_cuts['objective']:.4f}")
                         current_node.lp_objective = lp_result_after_cuts['objective']
                         current_node.lp_solution = lp_result_after_cuts['solution']
                         current_node.vbasis = lp_result_after_cuts.get('vbasis')
                         current_node.cbasis = lp_result_after_cuts.get('cbasis')
                    else:
                         logger.warning(f"LP re-solve failed with status {lp_result_after_cuts['status']}. Pruning node.")
                         continue
            # --- END FIX ---

            if self.incumbent_objective is not None:
                if (self.is_maximization and current_node.lp_objective <= self.incumbent_objective) or \
                   (not self.is_maximization and current_node.lp_objective >= self.incumbent_objective):
                    logger.info(f"Node {current_node.node_id} pruned by bound.")
                    continue

            if current_node.lp_solution and self._is_integer_feasible(current_node.lp_solution):
                is_new_best = self.incumbent_objective is None or \
                              (self.is_maximization and current_node.lp_objective > self.incumbent_objective) or \
                              (not self.is_maximization and current_node.lp_objective < self.incumbent_objective)
                if is_new_best:
                    self.incumbent_solution = current_node.lp_solution
                    self.incumbent_objective = current_node.lp_objective
                    logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")
                continue

            current_node.lp_solution['objective'] = current_node.lp_objective
            branch_var_name = self._get_branching_variable(current_node.lp_solution, current_node.local_constraints)
            if branch_var_name is None:
                logger.info(f"Node {current_node.node_id} has no fractional integer variables to branch on. Fathoming.")
                continue

            branch_val = current_node.lp_solution[branch_var_name]
            logger.info(f"Branching on variable {branch_var_name} with value {branch_val:.4f} from node {current_node.node_id}")

            for direction, sense, val in [('down', '<=', math.floor(branch_val)), ('up', '>=', math.ceil(branch_val))]:
                child_constraints = current_node.local_constraints + [(branch_var_name, sense, val)]
                child_lp_result = solve_lp_relaxation(self.problem, child_constraints)

                if child_lp_result['status'] == 'OPTIMAL':
                    degradation = abs(child_lp_result['objective'] - current_node.lp_objective)
                    self._update_pseudocosts(branch_var_name, direction, degradation)
                    
                    if self.incumbent_objective is not None and \
                       ((self.is_maximization and child_lp_result['objective'] <= self.incumbent_objective) or \
                       (not self.is_maximization and child_lp_result['objective'] >= self.incumbent_objective)):
                        logger.info(f"Child node pruned by bound upon creation.")
                        continue
                    
                    child_node = Node(node_id=self.node_counter, parent_id=current_node.node_id, local_constraints=child_constraints, lp_objective=child_lp_result['objective'], lp_solution=child_lp_result['solution'], status='SOLVED', vbasis=child_lp_result.get('vbasis'), cbasis=child_lp_result.get('cbasis'))
                    self.node_counter += 1
                    self.active_nodes.append(child_node)
                    logger.info(f"Child node {child_node.node_id} created. LP Obj: {child_lp_result['objective']:.4f}")
                else:
                    logger.info(f"Child node is {child_lp_result['status']}. Pruning.")

        logger.info("Branch and Bound solver finished.")
        if self.incumbent_solution:
            logger.info(f"Best solution found. Objective: {self.incumbent_objective:.4f}")
        else:
            logger.info("No integer-feasible solution found.")

        return self.incumbent_solution, self.incumbent_objective
