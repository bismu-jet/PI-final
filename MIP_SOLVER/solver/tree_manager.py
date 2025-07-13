import time
import yaml
import math
import gurobipy as gp
from typing import List, Dict, Optional, Tuple, Any

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
        
        # --- ROBUST INITIALIZATION ---
        # Determine the model sense ONCE and store it in our own attribute.
        self.is_maximization = self.problem.model.ModelSense == gp.GRB.MAXIMIZE
        model_sense = "MAXIMIZE" if self.is_maximization else "MINIMIZE"
        logger.info(f"Problem recognized as a {model_sense} problem.")
        
        logger.info("--- Starting Presolve Phase ---")
        presolve(self.problem)
        logger.info("--- Presolve Phase Finished ---")
        
        self.active_nodes: List[Node] = []
        self.incumbent_solution: Optional[Dict[str, float]] = None
        self.incumbent_objective: Optional[float] = None
        
        # Use our new attribute to set the initial bound correctly.
        self.global_best_bound: float = -math.inf if self.is_maximization else math.inf
        
        self.node_counter: int = 0
        self.optimality_gap = self.config['solver_params']['optimality_gap']
        self.time_limit_seconds = self.config['solver_params']['time_limit_seconds']

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
        Supports 'most_fractional' and 'strong_branching'.
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
                fractional_part = abs(val - round(val))
                if fractional_part > max_fractionality:
                    max_fractionality = fractional_part
                    best_var = var_name
            return best_var
        elif strategy == "strong_branching":
            best_var = None
            max_obj_change = -math.inf

            for var_name in fractional_vars:
                val = solution[var_name]
                floor_val = math.floor(val)
                ceil_val = math.ceil(val)

                # Trial branch: x <= floor(val)
                constraints_down = current_constraints + [(var_name, "<=", float(floor_val))]
                lp_result_down = solve_lp_relaxation(self.problem, constraints_down)

                # Trial branch: x >= ceil(val)
                constraints_up = current_constraints + [(var_name, ">=", float(ceil_val))]
                lp_result_up = solve_lp_relaxation(self.problem, constraints_up)

                obj_change = 0.0
                if lp_result_down["status"] == "OPTIMAL":
                    obj_change += abs(lp_result_down["objective"] - solution[var_name]) # Approximation of change
                if lp_result_up["status"] == "OPTIMAL":
                    obj_change += abs(lp_result_up["objective"] - solution[var_name]) # Approximation of change

                # If both branches are infeasible, this variable is a good candidate for branching
                if lp_result_down["status"] == "INFEASIBLE" and lp_result_up["status"] == "INFEASIBLE":
                    obj_change = math.inf # Prioritize variables that lead to infeasibility

                if obj_change > max_obj_change:
                    max_obj_change = obj_change
                    best_var = var_name
            return best_var
        else:
            logger.warning(f"Unsupported branching variable strategy: {strategy}")
            return None

    def solve(self):
        """
        Runs the Branch and Bound algorithm to solve the MIP problem.
        """
        logger.info("Starting Branch and Bound solver...")

        # Create the root Node (ID 0, no constraints)
        root_node = Node(
            node_id=self.node_counter,
            parent_id=None,
            local_constraints=[],
            lp_objective=None,
            lp_solution=None,
            status='PENDING'
        )
        self.node_counter += 1

        # Solve the root node's LP relaxation
        logger.info(f"Solving root node {root_node.node_id} LP relaxation...")
        lp_result = solve_lp_relaxation(self.problem, root_node.local_constraints)

        if lp_result['status'] == 'OPTIMAL':
            root_node.lp_objective = lp_result['objective']
            root_node.lp_solution = lp_result['solution']
            root_node.status = 'SOLVED'
            self.global_best_bound = root_node.lp_objective # Initial global best bound
            self.active_nodes.append(root_node)
            logger.info(f"Root node {root_node.node_id} solved. LP Objective: {root_node.lp_objective:.4f}")

            # Try to find an initial integer solution using the diving heuristic from the root LP solution
            initial_solution_from_heuristic = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints)
            if initial_solution_from_heuristic:
                # Evaluate the objective of the initial solution
                initial_obj_val = sum(initial_solution_from_heuristic.get(v.VarName, 0) * v.Obj for v in self.problem.model.getVars())

                self.incumbent_solution = initial_solution_from_heuristic
                self.incumbent_objective = initial_obj_val
                logger.info(f"Found initial incumbent solution via diving heuristic. Objective: {self.incumbent_objective:.4f}")
        elif lp_result['status'] == 'INFEASIBLE':
            root_node.status = 'PRUNED_INFEASIBLE'
            logger.info(f"Root node {root_node.node_id} is infeasible. Pruning.")
            return # No feasible solution
        else:
            logger.error(f"Root node {root_node.node_id} LP relaxation failed with status: {lp_result['status']}")
            return # Error or other unhandled status

        # Main Branch and Bound loop
        start_time = time.time()
        while self.active_nodes:
            # Check time limit
            if time.time() - start_time > self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break

            # Calculate and check optimality gap
            if self.incumbent_objective is not None and self.global_best_bound is not None:
                if abs(self.incumbent_objective) > 1e-9:
                    if self.is_maximization:
                        gap = (self.global_best_bound - self.incumbent_objective) / abs(self.incumbent_objective)
                    else: # MINIMIZE
                        gap = (self.incumbent_objective - self.global_best_bound) / abs(self.incumbent_objective)
                    
                    if gap <= self.optimality_gap:
                        logger.info(f"Optimality gap ({gap:.6f}) reached {self.optimality_gap}. Terminating solver.")
                        break

            # a. Select Node
            current_node: Optional[Node] = None
            node_selection_strategy = self.config['strategy']['node_selection']

            if node_selection_strategy == 'best_bound':
                if self.is_maximization:
                    current_node = max(self.active_nodes, key=lambda node: node.lp_objective)
                else:
                    current_node = min(self.active_nodes, key=lambda node: node.lp_objective)
            elif node_selection_strategy == 'depth_first':
                current_node = self.active_nodes.pop() 
            elif node_selection_strategy == 'best_estimate':
                if self.is_maximization:
                    current_node = max(self.active_nodes, key=lambda node: node.lp_objective if node.lp_objective is not None else -math.inf)
                else:
                    current_node = min(self.active_nodes, key=lambda node: node.lp_objective if node.lp_objective is not None else math.inf)
            else:
                logger.error(f"Unsupported node selection strategy: {strategy}")
                break

            if current_node is None:
                break 

            if node_selection_strategy != 'depth_first':
                self.active_nodes.remove(current_node)
            
            logger.info(f"Processing node {current_node.node_id}. LP Objective: {current_node.lp_objective:.4f}")

            # b. Process Node & c. Pruning/Fathoming
            # Check if the node's LP objective is worse than the current incumbent
            if self.incumbent_objective is not None:
                if self.is_maximization:
                    if current_node.lp_objective <= self.incumbent_objective: # Prune by bound
                        logger.info(f"Node {current_node.node_id} pruned by bound. LP Obj ({current_node.lp_objective:.4f}) <= Incumbent ({self.incumbent_objective:.4f})")
                        continue
                else: # MINIMIZE
                    if current_node.lp_objective >= self.incumbent_objective: # Prune by bound
                        logger.info(f"Node {current_node.node_id} pruned by bound. LP Obj ({current_node.lp_objective:.4f}) >= Incumbent ({self.incumbent_objective:.4f})")
                        continue

            # Check if the LP solution is integer-feasible
            if current_node.lp_solution and self._is_integer_feasible(current_node.lp_solution):
                logger.info(f"Node {current_node.node_id} found integer-feasible solution.")

                current_objective = current_node.lp_objective
                
                is_new_best = self.incumbent_objective is None or \
                              (self.is_maximization and current_objective > self.incumbent_objective) or \
                              (not self.is_maximization and current_objective < self.incumbent_objective)

                if is_new_best:
                    self.incumbent_solution = {k: round(v) for k, v in current_node.lp_solution.items()}
                    self.incumbent_objective = current_objective
                    logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")

                continue # Fathom this node

            # d. Branching
            branch_var_name = self._get_branching_variable(current_node.lp_solution, current_node.local_constraints)
            if branch_var_name is None:
                logger.info(f"Node {current_node.node_id} has no fractional integer variables to branch on. Fathoming.")
                continue

            branch_val = current_node.lp_solution[branch_var_name]
            branch_val_floor = math.floor(branch_val)
            branch_val_ceil = math.ceil(branch_val)

            logger.info(f"Branching on variable {branch_var_name} with value {branch_val:.4f} from node {current_node.node_id}")

            # Child A: x <= floor(val)
            child_a_constraints = current_node.local_constraints + [(branch_var_name, '<=', float(branch_val_floor))]
            child_a = Node(
                node_id=self.node_counter,
                parent_id=current_node.node_id,
                local_constraints=child_a_constraints,
                lp_objective=None,
                lp_solution=None,
                status='PENDING'
            )
            self.node_counter += 1

            # Child B: x >= ceil(val)
            child_b_constraints = current_node.local_constraints + [(branch_var_name, '>=', float(branch_val_ceil))]
            child_b = Node(
                node_id=self.node_counter,
                parent_id=current_node.node_id,
                local_constraints=child_b_constraints,
                lp_objective=None,
                lp_solution=None,
                status='PENDING'
            )
            self.node_counter += 1

            # Solve children LP relaxations
            for child_node in [child_a, child_b]:
                logger.info(f"Solving child node {child_node.node_id} LP relaxation...")
                child_lp_result = solve_lp_relaxation(self.problem, child_node.local_constraints)

                if child_lp_result['status'] == 'OPTIMAL':
                    child_node.lp_objective = child_lp_result['objective']
                    child_node.lp_solution = child_lp_result['solution']
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
