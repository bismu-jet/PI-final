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
        """
        Initializes the TreeManager, loading configuration and the MIP problem.

        Args:
            problem_path (str): Path to the MIP problem file.
            config_path (str): Path to the solver configuration file.
        """
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
        
        # --- NEW: Flag for hybrid strategy ---
        self.switched_to_best_bound = False
        # ---

        logger.info(f"Initialized TreeManager with problem: {problem_path} and config: {config_path}")

    def _is_integer_feasible(self, solution: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """
        Checks if a given solution is integer-feasible for all integer variables.
        """
        for var_name in self.problem.integer_variable_names:
            if var_name in solution:
                if abs(solution[var_name] - round(solution[var_name])) > tolerance:
                    return False
        return True

    def _get_branching_variable(self, solution: Dict[str, float], current_constraints: List[Tuple[str, str, float]]) -> Optional[str]:
        """
        Selects a variable to branch on based on the configured strategy.
        This version uses a more robust weighted score for strong branching.
        """
        strategy = self.config["strategy"]["branching_variable"]

        fractional_vars = []
        for var_name in self.problem.integer_variable_names:
            if var_name in solution:
                val = solution[var_name]
                if abs(val - round(val)) > 1e-6:
                    fractional_vars.append(var_name)

        if not fractional_vars:
            return None

        if strategy == "most_fractional":
            best_var = None
            max_fractionality = -1.0
            for var_name in fractional_vars:
                val = solution[var_name]
                fractional_part = 0.5 - abs(val - math.floor(val) - 0.5)
                if fractional_part > max_fractionality:
                    max_fractionality = fractional_part
                    best_var = var_name
            return best_var
        
        elif strategy == "strong_branching":
            best_var = None
            max_score = -1.0
            parent_obj = solution['objective']

            for var_name in fractional_vars:
                val = solution[var_name]
                floor_val = math.floor(val)
                ceil_val = math.ceil(val)
                fractional_part = val - floor_val

                constraints_down = current_constraints + [(var_name, "<=", float(floor_val))]
                lp_result_down = solve_lp_relaxation(self.problem, constraints_down)
                
                constraints_up = current_constraints + [(var_name, ">=", float(ceil_val))]
                lp_result_up = solve_lp_relaxation(self.problem, constraints_up)

                degradation_down = math.inf
                if lp_result_down["status"] == "OPTIMAL":
                    degradation_down = abs(lp_result_down["objective"] - parent_obj)

                degradation_up = math.inf
                if lp_result_up["status"] == "OPTIMAL":
                    degradation_up = abs(lp_result_up["objective"] - parent_obj)
                
                score = (1 - fractional_part) * degradation_down + fractional_part * degradation_up
                
                if score > max_score:
                    max_score = score
                    best_var = var_name
            
            logger.info(f"Strong branching choice: '{best_var}' with score {max_score:.4f}")
            return best_var
        else:
            logger.warning(f"Unsupported branching variable strategy: {strategy}")
            return None

    def _find_violated_pool_cuts(self, solution: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Searches the global cut pool for cuts that are violated by the given solution.
        """
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
        """
        Runs the Branch and Bound algorithm to solve the MIP problem.
        """
        logger.info("Starting Branch and Bound solver...")

        root_node = Node(
            node_id=self.node_counter,
            parent_id=None,
            local_constraints=[],
            lp_objective=None,
            lp_solution=None,
            status='PENDING'
        )
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

            initial_solution_from_heuristic = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints)
            if initial_solution_from_heuristic:
                initial_obj_val = sum(initial_solution_from_heuristic.get(v.VarName, 0) * v.Obj for v in self.problem.model.getVars())
                self.incumbent_solution = initial_solution_from_heuristic
                self.incumbent_objective = initial_obj_val
                logger.info(f"Found initial incumbent solution via diving heuristic. Objective: {self.incumbent_objective:.4f}")
        elif lp_result['status'] == 'INFEASIBLE':
            root_node.status = 'PRUNED_INFEASIBLE'
            logger.info(f"Root node {root_node.node_id} is infeasible. Pruning.")
            return None, None
        else:
            logger.error(f"Root node {root_node.node_id} LP relaxation failed with status: {lp_result['status']}")
            return None, None

        start_time = time.time()
        while self.active_nodes:
            if time.time() - start_time > self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break

            if self.incumbent_objective is not None and self.global_best_bound is not None:
                if abs(self.incumbent_objective) > 1e-9:
                    if self.is_maximization:
                        gap = (self.global_best_bound - self.incumbent_objective) / abs(self.incumbent_objective)
                    else:
                        gap = (self.incumbent_objective - self.global_best_bound) / abs(self.incumbent_objective)
                    
                    if gap <= self.optimality_gap:
                        logger.info(f"Optimality gap ({gap:.6f}) reached {self.optimality_gap}. Terminating solver.")
                        break

            # --- MODIFIED NODE SELECTION LOGIC ---
            current_node: Optional[Node] = None
            node_selection_strategy = self.config['strategy']['node_selection']

            if node_selection_strategy == 'best_bound':
                if self.is_maximization:
                    current_node = max(self.active_nodes, key=lambda node: node.lp_objective)
                else:
                    current_node = min(self.active_nodes, key=lambda node: node.lp_objective)
                self.active_nodes.remove(current_node)

            elif node_selection_strategy == 'depth_first':
                current_node = self.active_nodes.pop()
            
            elif node_selection_strategy == 'hybrid':
                # Switch to best_bound once the first incumbent is found
                if self.incumbent_solution is not None and not self.switched_to_best_bound:
                    logger.info("Incumbent found. Switching node selection strategy to Best-Bound.")
                    self.switched_to_best_bound = True
                
                if not self.switched_to_best_bound:
                    # Use Depth-First Search until an incumbent is found
                    current_node = self.active_nodes.pop()
                else:
                    # Use Best-Bound Search after an incumbent is found
                    if self.is_maximization:
                        current_node = max(self.active_nodes, key=lambda node: node.lp_objective)
                    else:
                        current_node = min(self.active_nodes, key=lambda node: node.lp_objective)
                    self.active_nodes.remove(current_node)
            
            else:
                logger.error(f"Unsupported node selection strategy: {node_selection_strategy}")
                break
            # --- END MODIFICATION ---

            if current_node is None:
                break 
            
            logger.info(f"Processing node {current_node.node_id}. LP Objective: {current_node.lp_objective:.4f}")
            
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
            
            if self.incumbent_objective is not None:
                if (self.is_maximization and current_node.lp_objective <= self.incumbent_objective) or \
                   (not self.is_maximization and current_node.lp_objective >= self.incumbent_objective):
                    logger.info(f"Node {current_node.node_id} pruned by bound.")
                    continue

            if current_node.lp_solution and self._is_integer_feasible(current_node.lp_solution):
                clean_integer_solution = {k: round(v) for k, v in current_node.lp_solution.items() if k in self.problem.integer_variable_names}
                for v_name, v_val in current_node.lp_solution.items():
                    if v_name not in clean_integer_solution:
                        clean_integer_solution[v_name] = v_val

                current_objective = sum(clean_integer_solution.get(v.VarName, 0) * v.Obj for v in self.problem.model.getVars())
                
                is_new_best = self.incumbent_objective is None or \
                              (self.is_maximization and current_objective > self.incumbent_objective) or \
                              (not self.is_maximization and current_objective < self.incumbent_objective)

                if is_new_best:
                    self.incumbent_solution = clean_integer_solution
                    self.incumbent_objective = current_objective
                    logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")

                continue

            # Add the current node's objective to its solution dict for strong branching
            current_node.lp_solution['objective'] = current_node.lp_objective
            branch_var_name = self._get_branching_variable(current_node.lp_solution, current_node.local_constraints)
            if branch_var_name is None:
                logger.info(f"Node {current_node.node_id} has no fractional integer variables to branch on. Fathoming.")
                continue

            branch_val = current_node.lp_solution[branch_var_name]
            branch_val_floor = math.floor(branch_val)
            branch_val_ceil = math.ceil(branch_val)

            logger.info(f"Branching on variable {branch_var_name} with value {branch_val:.4f} from node {current_node.node_id}")

            for sense, val in [('<=', float(branch_val_floor)), ('>=', float(branch_val_ceil))]:
                child_constraints = current_node.local_constraints + [(branch_var_name, sense, val)]
                child_node = Node(
                    node_id=self.node_counter,
                    parent_id=current_node.node_id,
                    local_constraints=child_constraints,
                    lp_objective=None,
                    lp_solution=None,
                    status='PENDING'
                )
                self.node_counter += 1

                logger.info(f"Solving child node {child_node.node_id} LP relaxation...")
                child_lp_result = solve_lp_relaxation(self.problem, child_node.local_constraints)

                if child_lp_result['status'] == 'OPTIMAL':
                    child_node.lp_objective = child_lp_result['objective']
                    child_node.lp_solution = child_lp_result['solution']
                    child_node.vbasis = child_lp_result.get('vbasis')
                    child_node.cbasis = child_lp_result.get('cbasis')
                    child_node.status = 'SOLVED'

                    if self.incumbent_objective is not None:
                        if (self.is_maximization and child_node.lp_objective <= self.incumbent_objective) or \
                           (not self.is_maximization and child_node.lp_objective >= self.incumbent_objective):
                            logger.info(f"Child node {child_node.node_id} pruned by bound upon creation.")
                            continue

                    self.active_nodes.append(child_node)
                    logger.info(f"Child node {child_node.node_id} solved. LP Objective: {child_node.lp_objective:.4f}")
                elif child_lp_result['status'] == 'INFEASIBLE':
                    logger.info(f"Child node {child_node.node_id} is infeasible. Pruning.")
                else:
                    logger.warning(f"Child node {child_node.node_id} LP relaxation failed with status: {child_lp_result['status']}. Pruning.")
                    child_node.status = 'PRUNED_ERROR'

        logger.info("Branch and Bound solver finished.")
        if self.incumbent_solution:
            logger.info(f"Optimal solution found. Objective: {self.incumbent_objective:.4f}")
        else:
            logger.info("No integer-feasible solution found.")

        return self.incumbent_solution, self.incumbent_objective
