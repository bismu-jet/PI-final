import time
import yaml
import math
import heapq 
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional, Tuple, Any

# --- Importing custom modules for the solver ---
# These modules contain specific parts of the Branch and Bound algorithm.
from solver.cuts import generate_all_cuts # For generating cutting planes.
from solver.problem import MIPProblem # To load and manage the MIP problem data.
from solver.node import Node # Represents a node in the Branch and Bound tree.
from solver.gurobi_interface import solve_lp_relaxation # Solves the LP relaxation using Gurobi.
from solver.heuristics import find_initial_solution, run_periodic_heuristics # For finding integer solutions.
from solver.utilities import setup_logger # Utility for logging solver progress.
from solver.presolve import presolve # For simplifying the problem before solving.

# --- Setup the logger ---
# Creates a logger object to print progress and debug information.
logger = setup_logger()

class TreeManager:
    """
    Manages the Branch and Bound (B&B) tree, implementing the core solver logic.
    This class orchestrates the entire solution process, from reading the problem
    to exploring the B&B tree and finding the optimal integer solution.
    """
    def __init__(self, problem_path: str, config_path: str):
        """
        Initializes the TreeManager.
        
        Args:
            problem_path (str): The file path to the MIP problem (e.g., .lp or .mps).
            config_path (str): The file path to the solver's YAML configuration file.
        """
        # Load solver configuration from a YAML file.
        with open(config_path, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Load the MIP problem using the custom MIPProblem class.
        self.problem: MIPProblem = MIPProblem(problem_path)
        
        # Check if the problem is for maximization or minimization.
        self.is_maximization: bool = self.problem.model.ModelSense == gp.GRB.MAXIMIZE
        model_sense: str = "MAXIMIZE" if self.is_maximization else "MINIMIZE"
        logger.info(f"Problem recognized as a {model_sense} problem.")
        
        # --- Presolve Phase ---
        # Apply presolve techniques to simplify the model before starting the B&B.
        logger.info("--- Starting Presolve Phase ---")
        presolve(self.problem)
        logger.info("--- Presolve Phase Finished ---")
        
        # List of active (unexplored) nodes in the B&B tree. Uses a min-heap.
        self.active_nodes: List[Node] = []
        # The best integer-feasible solution found so far.
        self.incumbent_solution: Optional[Dict[str, float]] = None
        # The objective value of the incumbent solution. Also known as the global lower/upper bound.
        self.incumbent_objective: Optional[float] = None
        
        # The global best bound from the LP relaxations of all active nodes.
        # For maximization, this is the highest LP objective. For minimization, the lowest.
        self.global_best_bound: float = -math.inf if self.is_maximization else math.inf
        
        # Counter for creating unique node IDs.
        self.node_counter: int = 0
        # The desired optimality gap to terminate the solver.
        self.optimality_gap: float = self.config['solver_params']['optimality_gap']
        # The maximum time allowed for the solver to run.
        self.time_limit_seconds: int = self.config['solver_params']['time_limit_seconds']

        # A pool to store globally valid cuts found during the search.
        self.cut_pool: List[Dict[str, Any]] = []
        
        # --- Pseudocost Initialization ---
        # Used for a smart branching variable selection strategy.
        self.pseudocosts: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Stores the global average degradation for branching up.
        self.global_pseudocost_up: Dict[str, float] = {'sum_degrad': 0.0, 'count': 0}
        # Stores the global average degradation for branching down.
        self.global_pseudocost_down: Dict[str, float] = {'sum_degrad': 0.0, 'count': 0}

        logger.info(f"Initialized TreeManager with problem: {problem_path} and config: {config_path}")

    def _update_incumbent(self, new_solution: Dict[str, float], new_objective: float) -> bool:
        """
        Updates the incumbent solution if a new, better integer solution is found.
        
        Args:
            new_solution (Dict[str, float]): The new integer-feasible solution.
            new_objective (float): The objective value of the new solution.
            
        Returns:
            bool: True if the incumbent was updated, False otherwise.
        """
        # Determine if the new solution is better than the current incumbent.
        is_new_best: bool = self.incumbent_objective is None or \
                      (self.is_maximization and new_objective > self.incumbent_objective) or \
                      (not self.is_maximization and new_objective < self.incumbent_objective)
        
        if is_new_best:
            # If this is the first incumbent, switch node selection to best-bound.
            if self.incumbent_objective is None:
                Node.switch_to_bb = True
                Node.is_maximization = self.is_maximization
                logger.info("--- First incumbent found. Switching node selection strategy to Best-Bound. ---")
                # Reorganize the list into a proper heap for best-bound selection.
                heapq.heapify(self.active_nodes)

            # Store the new best solution and its objective value.
            self.incumbent_solution = new_solution
            self.incumbent_objective = new_objective
            logger.info(f"New incumbent found! Objective: {self.incumbent_objective:.4f}")
            
            # Prune the tree: remove active nodes that are no longer promising.
            self.active_nodes = [
                n for n in self.active_nodes if self._is_promising(n)
            ]
            # Re-heapify the list after pruning nodes.
            heapq.heapify(self.active_nodes)
            return True
        return False

    def _is_promising(self, node: Node) -> bool:
        """
        Checks if a node is promising, i.e., if it could potentially lead to a
        better solution than the current incumbent.
        
        Args:
            node (Node): The node to check.
            
        Returns:
            bool: True if the node is promising, False otherwise.
        """
        # If no incumbent exists yet, any feasible node is promising.
        if self.incumbent_objective is None:
            return True
        # For maximization, the node's LP objective must be greater than the incumbent's.
        if self.is_maximization:
            # Use a small tolerance for floating-point comparisons.
            return node.lp_objective > self.incumbent_objective + 1e-6
        # For minimization, the node's LP objective must be less than the incumbent's.
        else:
            return node.lp_objective < self.incumbent_objective - 1e-6

    def _is_integer_feasible(self, solution: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """
        Checks if a solution is integer-feasible within a given tolerance.
        
        Args:
            solution (Dict[str, float]): The solution to check.
            tolerance (float): The tolerance for checking integrality.
            
        Returns:
            bool: True if all integer variables have integer values, False otherwise.
        """
        # Iterate through all variables that are supposed to be integers.
        for var_name in self.problem.integer_variable_names:
            if var_name in solution:
                # Check if the variable's value is close to a whole number.
                if abs(solution[var_name] - round(solution[var_name])) > tolerance:
                    return False
        return True

    def _update_pseudocosts(self, var_name: str, direction: str, degradation: float):
        """
        Updates the pseudocost information for a variable after branching.
        Pseudocosts measure how much the objective function degrades when forcing
        a fractional variable towards an integer value.
        
        Args:
            var_name (str): The name of the variable that was branched on.
            direction (str): 'up' or 'down', indicating the branching direction.
            degradation (float): The observed change in the objective function.
        """
        # Initialize pseudocost dictionary for the variable if it doesn't exist.
        if var_name not in self.pseudocosts:
            self.pseudocosts[var_name] = {
                'up': {'sum_degrad': 0.0, 'count': 0},
                'down': {'sum_degrad': 0.0, 'count': 0}
            }
        
        # Update the specific variable's pseudocost data.
        self.pseudocosts[var_name][direction]['sum_degrad'] += degradation
        self.pseudocosts[var_name][direction]['count'] += 1
        
        # Update the global pseudocost averages.
        if direction == 'up':
            self.global_pseudocost_up['sum_degrad'] += degradation
            self.global_pseudocost_up['count'] += 1
        else:
            self.global_pseudocost_down['sum_degrad'] += degradation
            self.global_pseudocost_down['count'] += 1
            
        logger.debug(f"Updated pseudocost for '{var_name}' ({direction}): degradation={degradation:.4f}")

    def _select_by_pseudocost(self, solution: Dict[str, float], fractional_vars: List[str]) -> str:
        """
        Selects the best fractional variable to branch on using pseudocost scoring.
        This is often more effective than simple strategies like "most fractional".
        
        Args:
            solution (Dict[str, float]): The current fractional LP solution.
            fractional_vars (List[str]): The list of variables with fractional values.
            
        Returns:
            str: The name of the selected variable to branch on.
        """
        best_var: Optional[str] = None
        max_score: float = -1.0

        # Calculate average degradation from global data (reliability fallback).
        avg_up: float = (self.global_pseudocost_up['sum_degrad'] / self.global_pseudocost_up['count']) \
                 if self.global_pseudocost_up['count'] > 0 else 1.0
        avg_down: float = (self.global_pseudocost_down['sum_degrad'] / self.global_pseudocost_down['count']) \
                   if self.global_pseudocost_down['count'] > 0 else 1.0
        
        # Evaluate each fractional variable.
        for var_name in fractional_vars:
            val: float = solution[var_name]
            frac_part: float = val - math.floor(val)
            
            # Get specific pseudocosts for this variable, or fall back to globals.
            var_info: Dict[str, Dict[str, float]] = self.pseudocosts.get(var_name, {'up': {'count': 0}, 'down': {'count': 0}})
            
            # Use specific pseudocost if available (reliable), otherwise use the global average.
            pc_down: float = (var_info['down']['sum_degrad'] / var_info['down']['count']) if var_info['down']['count'] > 0 else avg_down
            pc_up: float = (var_info['up']['sum_degrad'] / var_info['up']['count']) if var_info['up']['count'] > 0 else avg_up
            
            # Calculate the score: a weighted average of the expected degradation.
            score: float = (1 - frac_part) * pc_down + frac_part * pc_up
            
            # Keep track of the variable with the highest score.
            if score > max_score:
                max_score = score
                best_var = var_name
                
        logger.info(f"Pseudocost choice: '{best_var}' with score {max_score:.4f} (using reliability logic)")
        return best_var

    def _get_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        """
        Determines which fractional variable to branch on.
        
        Args:
            solution (Dict[str, float]): The fractional LP solution of the current node.
        
        Returns:
            Optional[str]: The name of the branching variable, or None if the solution is integer-feasible.
        """
        # Find all integer variables that currently have fractional values.
        fractional_vars: List[str] = [
            var_name for var_name in self.problem.integer_variable_names
            if var_name in solution and abs(solution[var_name] - round(solution[var_name])) > 1e-6
        ]

        # If there are no fractional variables, branching is not needed.
        if not fractional_vars:
            return None

        # Select the branching strategy based on the config file.
        strategy: str = self.config["strategy"]["branching_variable"]
        if strategy == "pseudocost":
            return self._select_by_pseudocost(solution, fractional_vars)
        else: # Default to "most fractional" (closest to 0.5)
            return max(fractional_vars, key=lambda v: 0.5 - abs(solution[v] - math.floor(solution[v]) - 0.5))

    def solve(self) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
        """
        The main Branch and Bound solver loop.
        
        Returns:
            A tuple containing the best solution dictionary and its objective value,
            or (None, None) if no solution is found.
        """
        logger.info("Starting Branch and Bound solver...")

        # --- Root Node Initialization ---
        # Create the first node (root) of the B&B tree.
        root_node: Node = Node(node_id=self.node_counter, parent_id=None, local_constraints=[], lp_objective=None, lp_solution=None, status='PENDING', depth=0)
        self.node_counter += 1

        # Solve the LP relaxation for the root node.
        logger.info(f"Solving root node {root_node.node_id} LP relaxation...")
        lp_result: Dict[str, Any] = solve_lp_relaxation(self.problem, root_node.local_constraints)

        # If the root LP is infeasible, the whole problem is infeasible.
        if lp_result['status'] != 'OPTIMAL':
            logger.error(f"Root node LP failed with status: {lp_result['status']}. Terminating.")
            return None, None

        # Store the results in the root node object.
        root_node.lp_objective = lp_result['objective']
        root_node.lp_solution = lp_result['solution']
        root_node.vbasis = lp_result.get('vbasis') # Basis info for warm starts.
        root_node.cbasis = lp_result.get('cbasis')
        root_node.status = 'SOLVED'
        # Add the solved root node to the list of active nodes.
        heapq.heappush(self.active_nodes, root_node)
        logger.info(f"Root node {root_node.node_id} solved. LP Objective: {root_node.lp_objective:.4f}")

        # --- Initial Heuristic ---
        # Try to find a first integer solution quickly using a heuristic.
        candidate_integer_solution: Optional[Dict[str, float]] = find_initial_solution(self.problem, root_node.lp_solution, root_node.local_constraints)
        if candidate_integer_solution:
            # If the heuristic finds a partial integer solution, try to complete it.
            fixed_vars_constraints: List[Tuple[str, str, float]] = [(v, '==', float(round(val))) for v, val in candidate_integer_solution.items() if v in self.problem.integer_variable_names]
            completion_lp_result: Dict[str, Any] = solve_lp_relaxation(self.problem, fixed_vars_constraints)
            if completion_lp_result['status'] == 'OPTIMAL':
                # If a valid full solution is found, update the incumbent.
                self._update_incumbent(completion_lp_result['solution'], completion_lp_result['objective'])
            else:
                logger.warning("Heuristic solution was not extendable to a feasible solution.")
        
        start_time: float = time.time()
        # --- Main B&B Loop ---
        # Continue as long as there are active nodes to explore.
        while self.active_nodes:
            # Check for termination due to time limit.
            if time.time() - start_time > self.time_limit_seconds:
                logger.info(f"Time limit of {self.time_limit_seconds} seconds reached. Terminating solver.")
                break
            
            # The best possible objective is the one from the best node in the heap.
            self.global_best_bound = self.active_nodes[0].lp_objective if self.active_nodes else (math.inf if not self.is_maximization else -math.inf)
            
            logger.info(f"--- Nodes: {len(self.active_nodes)}, Global Best Bound: {self.global_best_bound:.4f}, Incumbent: {self.incumbent_objective} ---")

            # Check for termination due to optimality gap.
            if self.incumbent_objective is not None:
                # Avoid division by zero.
                if abs(self.incumbent_objective) > 1e-9:
                    gap: float = abs(self.incumbent_objective - self.global_best_bound) / (abs(self.incumbent_objective) + 1e-9)
                    if gap <= self.optimality_gap:
                        logger.info(f"Optimality gap ({gap:.6f}) reached {self.optimality_gap}. Terminating solver.")
                        break

            # --- Node Selection ---
            # Select the most promising node from the heap to explore next.
            current_node: Node = heapq.heappop(self.active_nodes)
            
            # --- Pruning by Bound ---
            # (1) Discard the node if it cannot produce a better solution than the incumbent.
            if not self._is_promising(current_node):
                logger.debug(f"Node {current_node.node_id} pruned by bound.")
                continue

            # --- Fathoming by Integrality ---
            # (2) If the node's solution is integer-feasible, it's a potential new incumbent.
            if self._is_integer_feasible(current_node.lp_solution):
                logger.info(f"Node {current_node.node_id} is integer feasible. Fathoming.")
                self._update_incumbent(current_node.lp_solution, current_node.lp_objective)
                continue # Fathom this node as no further branching is needed.

            # --- Heuristics ---
            # (3) Periodically run heuristics to find new incumbent solutions.
            heuristic_freq: int = self.config['solver_params'].get('heuristic_frequency', 20)
            if self.incumbent_solution and self.node_counter % heuristic_freq == 1:
                heuristic_result: Optional[Dict[str, Any]] = run_periodic_heuristics(
                    problem=self.problem,
                    current_node_solution=current_node.lp_solution,
                    incumbent_solution=self.incumbent_solution,
                    config=self.config['solver_params']
                )
                if heuristic_result and self._update_incumbent(heuristic_result['solution'], heuristic_result['objective']):
                    # If the heuristic found a better incumbent, re-check if the current node is still promising.
                    if not self._is_promising(current_node):
                        logger.debug(f"Node {current_node.node_id} pruned by new incumbent from heuristic.")
                        continue
            
            # --- Cutting Planes ---
            # (4) Try to strengthen the LP relaxation of the current node by adding valid cuts.
            max_cut_rounds: int = 3
            cuts_this_node: List[Any] = []
            node_was_pruned_by_cuts: bool = False
            for round_num in range(max_cut_rounds):
                # Prepare LP result data needed for cut generation.
                lp_result_for_cuts: Dict[str, Any] = {
                    'solution': current_node.lp_solution,
                    'vbasis': current_node.vbasis,
                    'cbasis': current_node.cbasis,
                }
                
                # Generate new cutting planes based on the current fractional solution.
                new_cuts: List[Any] = generate_all_cuts(self.problem, lp_result_for_cuts, cuts_this_node, current_node.local_constraints)
                
                if not new_cuts:
                    logger.debug(f"Cut Pass {round_num + 1}: No new cuts found. Ending separation.")
                    break # Stop if no more cuts can be found.
                
                cuts_this_node.extend(new_cuts)
                logger.info(f"Cut Pass {round_num + 1}: Found {len(new_cuts)} cuts. Re-solving LP for node {current_node.node_id}.")
                
                # Re-solve the node's LP with the newly added cuts.
                lp_result_after_cuts: Dict[str, Any] = solve_lp_relaxation(self.problem, current_node.local_constraints, cuts=cuts_this_node)
                
                if lp_result_after_cuts['status'] == 'OPTIMAL':
                    # Check if the cuts improved the bound enough to prune the node.
                    temp_check_node: Node = Node(node_id=-1, parent_id=-1, lp_objective=lp_result_after_cuts['objective'], depth=0)
                    if not self._is_promising(temp_check_node):
                        logger.info("Node pruned by bound after cut application.")
                        node_was_pruned_by_cuts = True
                        break
                    
                    # Update the current node with the improved LP solution and objective.
                    current_node.lp_objective = lp_result_after_cuts['objective']
                    current_node.lp_solution = lp_result_after_cuts['solution']
                    current_node.vbasis = lp_result_after_cuts.get('vbasis')
                    current_node.cbasis = lp_result_after_cuts.get('cbasis')
                else:
                    # If adding cuts made the LP infeasible, the node can be pruned.
                    logger.warning(f"LP re-solve with cuts failed. Status: {lp_result_after_cuts['status']}. Pruning node.")
                    node_was_pruned_by_cuts = True
                    break
            
            if node_was_pruned_by_cuts:
                continue # Move to the next node in the tree.

            # --- Branching ---
            # (5) If the node is still fractional and promising, branch on a fractional variable.
            branch_var_name: Optional[str] = self._get_branching_variable(current_node.lp_solution)
            if branch_var_name is None:
                logger.warning(f"Node {current_node.node_id} is fractional but no branching variable found. Fathoming.")
                continue

            branch_val: float = current_node.lp_solution[branch_var_name]
            logger.info(f"Branching on variable {branch_var_name} with value {branch_val:.4f} from node {current_node.node_id}")

            # Create two child nodes: one for branching "down" and one for "up".
            for direction, sense, val in [('down', '<=', math.floor(branch_val)), ('up', '>=', math.ceil(branch_val))]:
                # Add the new branching constraint to the list of constraints for the child.
                child_constraints: List[Tuple] = current_node.local_constraints + [(branch_var_name, sense, val)]
                # Solve the LP relaxation for the new child node.
                child_lp_result: Dict[str, Any] = solve_lp_relaxation(self.problem, child_constraints)

                if child_lp_result['status'] == 'OPTIMAL':
                    # Before adding the child node, check if it's promising.
                    if not self._is_promising(Node(node_id=-1, parent_id=-1, lp_objective=child_lp_result['objective'], depth=0)):
                        continue # Prune by bound before adding to the tree.
                    
                    # Calculate objective degradation for pseudocost update.
                    degradation: float = abs(child_lp_result['objective'] - current_node.lp_objective)
                    self._update_pseudocosts(branch_var_name, direction, degradation)
                    
                    # Create the new child node and add it to the active nodes heap.
                    child_node: Node = Node(
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
                    # If the child LP is infeasible or unbounded, it can be pruned.
                    logger.debug(f"Child node is {child_lp_result['status']}. Pruning.")

        # --- Solver Termination ---
        logger.info("Branch and Bound solver finished.")
        if self.incumbent_solution:
            logger.info(f"Best solution found. Objective: {self.incumbent_objective:.4f}")
        else:
            logger.info("No integer-feasible solution found.")

        return self.incumbent_solution, self.incumbent_objective