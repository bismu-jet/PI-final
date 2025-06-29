import gurobipy as gp
from typing import Dict, Optional, List, Tuple
from solver.problem import MIPProblem
from solver.gurobi_interface import solve_lp_relaxation
from solver.utilities import setup_logger
import math

logger = setup_logger()

def find_initial_solution(problem: MIPProblem, current_lp_solution: Dict[str, float], current_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Implements a simple diving heuristic to find an initial integer-feasible solution.
    It tries to round fractional variables and solve subsequent LPs.

    Args:
        problem (MIPProblem): The MIPProblem instance.
        current_lp_solution (Dict[str, float]): The LP solution from which to start diving.
        current_constraints (List[Tuple[str, str, float]]): Constraints active at the current node.

    Returns:
        Optional[Dict[str, float]]: An initial integer-feasible solution if found, otherwise None.
    """
    # Check if the current solution is already integer-feasible
    is_integer_feasible = True
    fractional_vars = []
    for var_name in problem.integer_variable_names:
        if var_name in current_lp_solution:
            val = current_lp_solution[var_name]
            if abs(val - round(val)) > 1e-6:
                is_integer_feasible = False
                fractional_vars.append(var_name)

    if is_integer_feasible:
        logger.info("Diving heuristic found an integer-feasible solution.")
        return current_lp_solution

    if not fractional_vars:
        return None # No fractional variables to branch on, but not integer feasible (shouldn't happen if check above is correct)

    # Select a fractional variable to dive on (e.g., the first one found)
    branch_var_name = fractional_vars[0]
    branch_val = current_lp_solution[branch_var_name]

    # Try rounding down
    logger.debug(f"Diving: Trying {branch_var_name} <= {math.floor(branch_val)}")
    constraints_down = current_constraints + [(branch_var_name, '<=', float(math.floor(branch_val)))]
    lp_result_down = solve_lp_relaxation(problem, constraints_down)

    if lp_result_down['status'] == 'OPTIMAL':
        solution_down = find_initial_solution(problem, lp_result_down['solution'], constraints_down)
        if solution_down:
            return solution_down

    # Try rounding up
    logger.debug(f"Diving: Trying {branch_var_name} >= {math.ceil(branch_val)}")
    constraints_up = current_constraints + [(branch_var_name, '>=', float(math.ceil(branch_val)))]
    lp_result_up = solve_lp_relaxation(problem, constraints_up)

    if lp_result_up['status'] == 'OPTIMAL':
        solution_up = find_initial_solution(problem, lp_result_up['solution'], constraints_up)
        if solution_up:
            return solution_up

    return None
