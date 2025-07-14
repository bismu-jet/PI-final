import time
import yaml
import math
import heapq 
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional, Tuple, Any

from solver.cuts import generate_all_cuts
from solver.problem import MIPProblem
from solver.node import Node
from solver.gurobi_interface import solve_lp_relaxation
from solver.heuristics import find_initial_solution, run_periodic_heuristics
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
        
        self.pseudocosts: Dict[str, Dict] = {}
        self.global_pseudocost_up = {'sum_degrad': 0.0, 'count': 0}
        self.global_pseudocost_down = {'sum_degrad': 0.0, 'count': 0}

        logger.info(f"Initialized TreeManager with problem: {problem_path} and config: {config_path}")

    def _update_incumbent(self, new_solution: Dict, new_objective: float) -> bool:
        is_new_best = self.incumbent_objective is None or \
                      (self.is_maximization and new_objective > self.incumbent_objective) or \
                      (not self.is_maximization and new_objective < self.incumbent_objective)
        
        if is_new_best:
            if self.incumbent_objective is None:
                Node.switch_to_bb = True
                Node.is_maximization = self.is_maximization
                logger.info("--- First incumbent found. Switching node selection strategy to Best-Bound. ---")
                heapq.heapify(self.active_nodes)

            self.incumbent_solution = new_solution
            self.incumbent_objective = new_objective
            logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")
            self.active_nodes = [
                n for n in self.active_nodes if self._is_promising(n)
            ]
            heapq.heapify(self.active_nodes)
            return True
        return False

    def _is_promising(self, node: Node) -> bool:
        if self.incumbent_objective is None:
            return True
        if self.is_maximization:
            return node.lp_objective > self.incumbent_objective + 1e-6
        else:
            return node.lp_objective < self.incumbent_objective - 1e-6

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
        
        if direction == 'up':
            self.global_pseudocost_up['sum_degrad'] += degradation
            self.global_pseudocost_up['count'] += 1
        else:
            self.global_pseudocost_down['sum_degrad'] += degradation
            self.global_pseudocost_down['count'] += 1
            
        logger.debug(f"Updated pseudocost for '{var_name}' ({direction}): degradation={degradation:.4f}")

    def _select_by_pseudocost(self, solution: Dict[str, float], fractional_vars: List[str]) -> str:
        best_var = None
        max_score = -1.0

        avg_up = (self.global_pseudocost_up['sum_degrad'] / self.global_pseudocost_up['count']) \
                 if self.global_pseudocost_up['count'] > 0 else 1.0
        avg_down = (self.global_pseudocost_down['sum_degrad'] / self.global_pseudocost_down['count']) \
                   if self.global_pseudocost_down['count'] > 0 else 1.0
        
        for var_name in fractional_vars:
            val = solution[var_name]
            frac_part = val - math.floor(val)
            
            var_info = self.pseudocosts.get(var_name, {'up': {'count': 0}, 'down': {'count': 0}})
            
            if var_info['down']['count'] > 0:
                pc_down = var_info['down']['sum_degrad'] / var_info['down']['count']
            else:
                pc_down = avg_down

            if var_info['up']['count'] > 0:
                pc_up = var_info['up']['sum_degrad'] / var_info['up']['count']
            else:
                pc_up = avg_up
            
            score = (1 - frac_part) * pc_down + frac_part * pc_up
            
            if score > max_score:
                max_score = score
                best_var = var_name
                
        logger.info(f"Pseudocost choice: '{best_var}' with score {max_score:.4f} (using reliability logic)")
        return best_var

    def _get_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        fractional_vars = [
            var_name for var_name in self.problem.integer_variable_names
            if var_name in solution and abs(solution[var_name] - round(solution[var_name])) > 1e-6
        ]

        if not fractional_vars:
            return None

        strategy = self.config["strategy"]["branching_variable"]
        if strategy == "pseudocost":
            return self._select_by_pseudocost(solution, fractional_vars)
        else:
            return max(fractional_vars, key=lambda v: 0.5 - abs(solution[v] - math.floor(solution[v]) - 0.5))

    def solve(self):
        logger.info("Starting Branch and Bound solver...")

        root_node = Node(node_id=self.node_counter, parent_id=None, local_constraints=[], lp_objective=None, lp_solution=None, status='PENDING', depth=0)
        self.node_counter += 1

        logger.info(f"Solving root node {root_node.node_id} LP relaxation...")
        lp_result = solve_lp_relaxation(self.problem, root_node.local_constraints)

        if lp_result['status'] != 'OPTIMAL':
            logger.error(f"Root node LP failed with status: {lp_result['status']}. Terminating.")
            return None, None

        root_node.lp_objective = lp_result['objective']
        root_node.lp_solution = lp_result['solution']
        root_node.vbasis = lp_result.get('vbasis')
        root_node.cbasis = lp_result.get('cbasis')
        root_node.status = 'SOLVED'
        heapq.heappush(self.active_nodes, root_node)
        logger.info(f"Root node {root_node.node_id} solved. LP Objective: {root_node.lp_objective:.4f}")

        candidate_integer_solution = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints)
        if candidate_integer_solution:
            fixed_vars_constraints = [(v, '==', float(round(val))) for v, val in candidate_integer_solution.items() if v in self.problem.integer_variable_names]
            completion_lp_result = solve_lp_relaxation(self.problem, fixed_vars_constraints)
            if completion_lp_result['status'] == 'OPTIMAL':
                self._update_incumbent(completion_lp_result['solution'], completion_lp_result['objective'])
            else:
                logger.warning("Heuristic solution was not extendable to a feasible solution.")
        
        start_time = time.time()
        while self.active_nodes:
            if time.time() - start_time > self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break
            
            self.global_best_bound = self.active_nodes[0].lp_objective if self.active_nodes else (math.inf if not self.is_maximization else -math.inf)
            
            logger.info(f"--- Nodes: {len(self.active_nodes)}, Global Best Bound: {self.global_best_bound:.4f}, Incumbent: {self.incumbent_objective} ---")

            if self.incumbent_objective is not None:
                if abs(self.incumbent_objective) > 1e-9:
                    gap = abs(self.incumbent_objective - self.global_best_bound) / (abs(self.incumbent_objective) + 1e-9)
                    if gap <= self.optimality_gap:
                        logger.info(f"Optimality gap ({gap:.6f}) reached {self.optimality_gap}. Terminating solver.")
                        break

            current_node = heapq.heappop(self.active_nodes)
            
            # --- AJUSTE NA LÓGICA PRINCIPAL ---
            # 1. Prune by bound (verificação mais barata)
            if not self._is_promising(current_node):
                logger.debug(f"Node {current_node.node_id} pruned by bound.")
                continue

            # 2. Verifica se a solução do nó é inteira
            if self._is_integer_feasible(current_node.lp_solution):
                logger.info(f"Node {current_node.node_id} is integer feasible. Fathoming.")
                self._update_incumbent(current_node.lp_solution, current_node.lp_objective)
                continue

            # --- A partir daqui, o nó é fracionário e promissor ---

            # 3. Tenta melhorar a solução com heurísticas periódicas
            heuristic_freq = self.config['solver_params'].get('heuristic_frequency', 20)
            if self.incumbent_solution and self.node_counter % heuristic_freq == 1:
                heuristic_result = run_periodic_heuristics(
                    problem=self.problem,
                    current_node_solution=current_node.lp_solution,
                    incumbent_solution=self.incumbent_solution,
                    config=self.config['solver_params']
                )
                if heuristic_result and self._update_incumbent(heuristic_result['solution'], heuristic_result['objective']):
                    if not self._is_promising(current_node):
                        logger.debug(f"Node {current_node.node_id} pruned by new incumbent from heuristic.")
                        continue
            
            # 4. Tenta fortalecer o nó com cortes
            max_cut_rounds = 3
            cuts_this_node = []
            node_was_pruned_by_cuts = False
            for round_num in range(max_cut_rounds):
                lp_result_for_cuts = {
                    'solution': current_node.lp_solution,
                    'vbasis': current_node.vbasis,
                    'cbasis': current_node.cbasis,
                }
                
                new_cuts = generate_all_cuts(self.problem, lp_result_for_cuts, cuts_this_node, current_node.local_constraints)
                
                if not new_cuts:
                    logger.debug(f"Cut Pass {round_num + 1}: No new cuts found. Ending separation.")
                    break
                
                cuts_this_node.extend(new_cuts)
                logger.info(f"Cut Pass {round_num + 1}: Found {len(new_cuts)} cuts. Re-solving LP for node {current_node.node_id}.")
                
                lp_result_after_cuts = solve_lp_relaxation(self.problem, current_node.local_constraints, cuts=cuts_this_node)
                
                if lp_result_after_cuts['status'] == 'OPTIMAL':
                    temp_check_node = Node(node_id=-1, parent_id=-1, lp_objective=lp_result_after_cuts['objective'], depth=0)
                    if not self._is_promising(temp_check_node):
                        logger.info("Node pruned by bound after cut application.")
                        node_was_pruned_by_cuts = True
                        break
                    
                    current_node.lp_objective = lp_result_after_cuts['objective']
                    current_node.lp_solution = lp_result_after_cuts['solution']
                    current_node.vbasis = lp_result_after_cuts.get('vbasis')
                    current_node.cbasis = lp_result_after_cuts.get('cbasis')
                else:
                    logger.warning(f"LP re-solve with cuts failed. Status: {lp_result_after_cuts['status']}. Pruning node.")
                    node_was_pruned_by_cuts = True
                    break
            
            if node_was_pruned_by_cuts:
                continue

            # 5. Se o nó ainda for viável, faz o branch
            branch_var_name = self._get_branching_variable(current_node.lp_solution)
            if branch_var_name is None:
                logger.warning(f"Node {current_node.node_id} is fractional but no branching variable found. Fathoming.")
                continue

            branch_val = current_node.lp_solution[branch_var_name]
            logger.info(f"Branching on variable {branch_var_name} with value {branch_val:.4f} from node {current_node.node_id}")

            for direction, sense, val in [('down', '<=', math.floor(branch_val)), ('up', '>=', math.ceil(branch_val))]:
                child_constraints = current_node.local_constraints + [(branch_var_name, sense, val)]
                child_lp_result = solve_lp_relaxation(self.problem, child_constraints)

                if child_lp_result['status'] == 'OPTIMAL':
                    if not self._is_promising(Node(node_id=-1, parent_id=-1, lp_objective=child_lp_result['objective'], depth=0)):
                        continue
                    
                    degradation = abs(child_lp_result['objective'] - current_node.lp_objective)
                    self._update_pseudocosts(branch_var_name, direction, degradation)
                    
                    child_node = Node(
                        node_id=self.node_counter, 
                        parent_id=current_node.node_id, 
                        local_constraints=child_constraints, 
                        lp_objective=child_lp_result['objective'], 
                        lp_solution=child_lp_result['solution'], 
                        status='SOLVED', 
                        vbasis=child_lp_result.get('vbasis'), 
                        cbasis=child_lp_result.get('cbasis'), 
                        depth=current_node.depth + 1
                    )
                    self.node_counter += 1
                    heapq.heappush(self.active_nodes, child_node)
                else:
                    logger.debug(f"Child node is {child_lp_result['status']}. Pruning.")

        logger.info("Branch and Bound solver finished.")
        if self.incumbent_solution:
            logger.info(f"Best solution found. Objective: {self.incumbent_objective:.4f}")
        else:
            logger.info("No integer-feasible solution found.")

        return self.incumbent_solution, self.incumbent_objective