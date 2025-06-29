import time
import yaml
import math
import logging # Import the logging module
import gurobipy as gp
from typing import List, Dict, Optional, Tuple

from solver.problem import MIPProblem
from solver.node import Node
from solver.gurobi_interface import solve_lp_relaxation
from solver.heuristics import find_initial_solution

# --- THIS IS THE CORRECT WAY TO GET THE LOGGER ---
# We get the logger instance that was already configured in main.py.
# We do NOT call setup_logger() here.
logger = logging.getLogger(__name__)
# ----------------------------------------------------

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
        self.active_nodes: List[Node] = []
        self.incumbent_solution: Optional[Dict[str, float]] = None
        self.incumbent_objective: Optional[float] = None
        self.global_best_bound: float = -math.inf if self.problem.model.ModelSense == gp.GRB.MAXIMIZE else math.inf
        self.node_counter: int = 0
        self.optimality_gap = self.config['solver_params']['optimality_gap']
        self.time_limit_seconds = self.config['solver_params']['time_limit_seconds']
        logger.info(f"Initialized TreeManager for problem: {problem_path}")

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
        Runs the full Branch and Bound algorithm to solve the MIP problem.
        This version correctly manages and passes the time limit to all LP solves.
        """
        logger.info("Starting Branch and Bound algorithm...")
        start_time = time.time()  # Start the master clock immediately

        # --- Step 1: Solve the Root Node ---
        # Inside the solve() method
        root_node = Node(
            node_id=self.node_counter,
            parent_id=None,
            local_constraints=[],
            lp_objective=None,      # <-- Restore this line
            lp_solution=None,       # <-- Restore this line
            status='PENDING'        # <-- Restore this line
        )
        self.node_counter += 1

        logger.info(f"Solving root node {root_node.node_id}...")
        # For the root solve, we pass the full time limit from the config.
        lp_result = solve_lp_relaxation(self.problem, root_node.local_constraints, time_limit=self.time_limit_seconds)

        # Check if the root solve was successful
        if lp_result.get('status') not in ['OPTIMAL', 'TIME_LIMIT']:
            logger.error(f"Root node LP failed. Status: {lp_result.get('status')}. Aborting.")
            return None, None
        
        # Even if time limit was hit, we might have a valid bound to start with.
        if lp_result.get('objective') is None:
             logger.error(f"Root node solve did not produce a valid objective. Aborting.")
             return None, None

        root_node.lp_objective = lp_result['objective']
        root_node.lp_solution = lp_result['solution']
        root_node.status = 'SOLVED'
        self.global_best_bound = root_node.lp_objective
        self.active_nodes.append(root_node)
        logger.info(f"Root node solved. Best Possible Bound: {self.global_best_bound:.4f}")

        # --- Step 2: Heuristic for Initial Solution ---
        if root_node.lp_solution:
            remaining_time = self.time_limit_seconds - (time.time() - start_time)
            heuristic_solution = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints, time_limit=remaining_time)
            if heuristic_solution:
                obj_expr = self.problem.model.getObjective()
                self.incumbent_objective = sum(
                    heuristic_solution.get(v.VarName, 0) * v.Obj 
                    for v in self.problem.model.getVars()
                )
                self.incumbent_solution = heuristic_solution
                logger.info(f"Heuristic found an initial integer solution! New Incumbent Objective: {self.incumbent_objective:.4f}")

        # --- Step 3: Main Branch and Bound Loop ---
        while self.active_nodes:
            # --- Termination Checks ---
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break

            if self.incumbent_objective is not None and abs(self.incumbent_objective) > 1e-9:
                gap = abs(self.global_best_bound - self.incumbent_objective) / abs(self.incumbent_objective)
                if gap <= self.optimality_gap:
                    logger.info(f"Optimality gap ({gap:.6f}) reached target ({self.optimality_gap}). Terminating solver.")
                    break
            
            # --- Node Selection (Your existing logic is fine here) ---
            node_selection_strategy = self.config['strategy']['node_selection']
            if node_selection_strategy == 'depth_first':
                current_node = self.active_nodes.pop()
            else:  # Default to 'best_bound'
                key_func = lambda n: n.lp_objective or (-math.inf if self.problem.model.ModelSense == gp.GRB.MAXIMIZE else math.inf)
                if self.problem.model.ModelSense == gp.GRB.MAXIMIZE:
                    current_node = max(self.active_nodes, key=key_func)
                else:
                    current_node = min(self.active_nodes, key=key_func)
                self.active_nodes.remove(current_node)
            
            # --- Enhanced Logging ---
            inc_obj_str = f"{self.incumbent_objective:.2f}" if self.incumbent_objective is not None else "None"
            logger.info(
                f"Processing Node {current_node.node_id} "
                f"[Obj: {current_node.lp_objective:.2f}, "
                f"Best Int: {inc_obj_str}, "
                f"Nodes Left: {len(self.active_nodes)}]"
            )

            # b. Process Node & c. Pruning/Fathoming
            # Check if the node's LP objective is worse than the current incumbent
            if self.incumbent_objective is not None:
                if self.problem.model.ModelSense == gp.GRB.MAXIMIZE:
                    if current_node.lp_objective <= self.incumbent_objective: # Prune by bound
                        logger.info(f"Node {current_node.node_id} pruned by bound. LP Obj ({current_node.lp_objective:.4f}) <= Incumbent ({self.incumbent_objective:.4f})")
                        current_node.status = 'PRUNED_BY_BOUND'
                        continue
                else: # MINIMIZE
                    if current_node.lp_objective >= self.incumbent_objective: # Prune by bound
                        logger.info(f"Node {current_node.node_id} pruned by bound. LP Obj ({current_node.lp_objective:.4f}) >= Incumbent ({self.incumbent_objective:.4f})")
                        current_node.status = 'PRUNED_BY_BOUND'
                        continue

            # Check if the LP solution is integer-feasible
            if current_node.lp_solution and self._is_integer_feasible(current_node.lp_solution):
                logger.info(f"Node {current_node.node_id} found integer-feasible solution.")
                # Calculate actual objective for this integer solution
                # This requires evaluating the original model's objective with the integer solution
                # For simplicity, using LP objective as incumbent objective for now. This needs refinement.
                current_objective = current_node.lp_objective

                if self.incumbent_objective is None or \
                   (self.problem.model.ModelSense == gp.GRB.MAXIMIZE and current_objective > self.incumbent_objective) or \
                   (self.problem.model.ModelSense == gp.GRB.MINIMIZE and current_objective < self.incumbent_objective):
                    self.incumbent_solution = current_node.lp_solution
                    self.incumbent_objective = current_objective
                    logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")

                current_node.status = 'FATHOMED'
                continue # Fathom this node

            # --- Branching ---
            remaining_time = self.time_limit_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                logger.info("No time remaining for child nodes. Terminating.")
                break

            branch_var_name = self._get_branching_variable(current_node.lp_solution, current_node.local_constraints)
            if not branch_var_name:
                continue

            branch_val = current_node.lp_solution[branch_var_name]
            
            for sense, bound in [('<=', math.floor(branch_val)), ('>=', math.ceil(branch_val))]:
                child_constraints = current_node.local_constraints + [(branch_var_name, sense, bound)]
                child_node = Node(
                    node_id=self.node_counter,
                    parent_id=current_node.node_id,
                    local_constraints=child_constraints,
                    lp_objective=None,   
                    lp_solution=None,       
                    status='PENDING'       
                )
                self.node_counter += 1

                # PASS THE CALCULATED REMAINING TIME TO THE CHILD SOLVE
                child_lp_result = solve_lp_relaxation(self.problem, child_node.local_constraints, time_limit=remaining_time)

                if child_lp_result.get('status') in ['OPTIMAL', 'TIME_LIMIT'] and child_lp_result.get('objective') is not None:
                    child_node.lp_objective = child_lp_result['objective']
                    child_node.lp_solution = child_lp_result['solution']
                    child_node.status = 'SOLVED'
                    self.active_nodes.append(child_node)
                else:
                    logger.debug(f"Child node {child_node.node_id} pruned due to infeasibility or solver error.")

        # --- Final Summary ---
        logger.info("Branch and Bound solver finished.")
        if self.incumbent_solution:
            logger.info(f"Final incumbent solution found. Objective: {self.incumbent_objective:.4f}")
        else:
            logger.info("No integer-feasible solution found.")
        
        return self.incumbent_solution, self.incumbent_objective