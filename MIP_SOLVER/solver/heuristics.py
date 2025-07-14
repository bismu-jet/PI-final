import gurobipy as gp
from gurobipy import GRB
import math
from typing import List, Dict, Optional, Tuple, Any

# --- Importing custom modules for the solver ---
from solver.problem import MIPProblem
from solver.gurobi_interface import solve_lp_relaxation, solve_lp_with_custom_objective, solve_sub_mip
from solver.utilities import setup_logger

# --- Setup the logger ---
# Creates a logger object to print progress and debug information for the heuristics.
logger = setup_logger()


def _rins_heuristic(problem: MIPProblem,
                    incumbent_solution: Dict[str, float],
                    current_lp_solution: Dict[str, float],
                    time_limit: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Implements the Relaxation Induced Neighborhood Search (RINS) Heuristic. 🔎

    RINS explores the neighborhood of the current best integer solution (the incumbent).
    The key idea is that if an integer variable has the same value in both the incumbent
    and the current LP relaxation solution, it's likely to be correct. This heuristic
    fixes these "agreed-upon" variables and solves a smaller sub-MIP on the
    remaining (free) variables to find a potentially better solution.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        incumbent_solution (Dict[str, float]): The best integer solution found so far.
        current_lp_solution (Dict[str, float]): The solution from the current node's LP relaxation.
        time_limit (float): The time limit for solving the sub-MIP.

    Returns:
        Optional[Dict[str, Any]]: A solution dictionary if a new one is found, otherwise None.
    """
    logger.info("Attempting to find solution with RINS Heuristic...")

    vars_to_fix: Dict[str, float] = {}
    # Identify the integer variables to fix based on agreement.
    for var_name in problem.integer_variable_names:
        incumbent_val: Optional[float] = incumbent_solution.get(var_name)
        lp_val: Optional[float] = current_lp_solution.get(var_name)

        # A comparison is only possible if the variable exists in both solutions.
        if incumbent_val is None or lp_val is None:
            continue

        # If the rounded integer values match, we fix the variable.
        if abs(round(incumbent_val) - round(lp_val)) < 1e-6:
            vars_to_fix[var_name] = round(incumbent_val)

    # If all integer variables are fixed, the sub-MIP is trivial and we can skip it.
    if len(vars_to_fix) == len(problem.integer_variable_names):
        logger.debug("RINS skipped: All integer variables are already fixed to the same values.")
        return None

    logger.info(f"RINS: Fixing {len(vars_to_fix)} integer variables and solving sub-MIP.")

    # Solve the smaller MIP with the fixed variables.
    sub_mip_result: Dict[str, Any] = solve_sub_mip(problem, vars_to_fix, time_limit)

    if sub_mip_result['status'] == 'FEASIBLE':
        logger.info(f"RINS found a feasible solution with objective: {sub_mip_result['objective']:.4f}")
        return sub_mip_result
    else:
        logger.info("RINS did not find a new solution.")
        return None


def _diving_heuristic(problem: MIPProblem,
                      initial_lp_solution: Dict[str, float],
                      initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Tries to find an integer-feasible solution using a simple Diving Heuristic. 🏊

    This heuristic works by iteratively "diving" towards an integer solution. In each step,
    it identifies the fractional variable that is closest to an integer value, fixes it to
    that integer value, and re-solves the LP. This process repeats until all integer
    variables are fixed or the subproblem becomes infeasible.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        initial_lp_solution (Dict[str, float]): The starting LP solution (usually from the root node).
        initial_constraints (List[Tuple[str, str, float]]): The starting constraints.

    Returns:
        Optional[Dict[str, float]]: A feasible integer solution if found, otherwise None.
    """
    logger.info("Attempting to find solution with Diving Heuristic...")

    current_solution: Dict[str, float] = initial_lp_solution.copy()
    current_constraints: List[Tuple[str, str, float]] = initial_constraints.copy()

    # We iterate a limited number of times to prevent getting stuck.
    for dive_iteration in range(problem.model.NumIntVars * 2):
        # Find all variables that are currently fractional.
        fractional_vars: List[Tuple[str, float]] = []
        for var_name in problem.integer_variable_names:
            if var_name in current_solution:
                val: float = current_solution[var_name]
                if abs(val - round(val)) > 1e-6:
                    # Calculate distance to the nearest integer (a value from 0 to 0.5).
                    distance: float = 0.5 - abs(val - math.floor(val) - 0.5)
                    fractional_vars.append((var_name, distance))

        # If there are no fractional variables, we have found an integer solution.
        if not fractional_vars:
            logger.info(f"Diving heuristic successful after {dive_iteration} dives.")
            # Return the rounded integer solution.
            return {v_name: round(v_val) for v_name, v_val in current_solution.items()}

        # Sort variables by their distance to the nearest integer, ascending.
        fractional_vars.sort(key=lambda x: x[1])
        # Select the variable closest to an integer to fix next.
        var_to_fix, _ = fractional_vars[0]

        val_to_fix: float = current_solution[var_to_fix]
        rounded_val: float = round(val_to_fix)

        logger.debug(f"Dive {dive_iteration}: Fixing '{var_to_fix}' from {val_to_fix:.4f} to {rounded_val}")

        # Add the new fixing constraint and re-solve the LP.
        current_constraints.append((var_to_fix, '==', float(rounded_val)))
        lp_result: Dict[str, Any] = solve_lp_relaxation(problem, current_constraints)

        if lp_result['status'] == 'OPTIMAL':
            current_solution = lp_result['solution']
        else:
            # If the LP becomes infeasible, this dive path has failed.
            logger.info(f"Diving heuristic failed at dive {dive_iteration}: subproblem became {lp_result['status']}.")
            return None

    logger.warning("Diving heuristic exceeded max iterations.")
    return None


def _feasibility_pump(problem: MIPProblem,
                      initial_lp_solution: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Tries to find an integer-feasible solution using a Feasibility Pump heuristic. ⛽

    This heuristic iterates between two main steps:
    1. It takes the current LP solution (`x_lp`) and rounds it to the nearest
       integer solution (`x_int`), which is likely infeasible.
    2. It then solves a new LP whose objective is to minimize the distance between a
       new feasible LP solution and the infeasible integer point `x_int`.
    This process "pumps" the solution towards a state where it is both integer and feasible.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        initial_lp_solution (Dict[str, float]): The starting LP solution.

    Returns:
        Optional[Dict[str, float]]: A feasible integer solution if found, otherwise None.
    """
    logger.info("Attempting to find solution with Feasibility Pump...")

    x_lp: Dict[str, float] = initial_lp_solution.copy()

    for pump_iteration in range(20):
        # Step 1: Round the current LP solution to get an integer point.
        x_int: Dict[str, float] = {var_name: round(val) for var_name, val in x_lp.items()}

        # Step 2: Define a new objective to find an LP solution closest to x_int.
        objective_coeffs: Dict[str, float] = {}
        for var_name in problem.integer_variable_names:
            # This logic sets up an L1-norm distance minimization objective.
            if x_int.get(var_name, 0) > 0.5: # If x_int is 1, we want x_lp to be 1.
                objective_coeffs[var_name] = -1.0
            else: # If x_int is 0, we want x_lp to be 0.
                objective_coeffs[var_name] = 1.0

        # Solve the LP with this custom "distance-minimizing" objective.
        lp_result: Dict[str, Any] = solve_lp_with_custom_objective(problem, objective_coeffs)

        if lp_result['status'] != 'OPTIMAL':
            logger.info(f"Feasibility Pump failed at iteration {pump_iteration}: distance LP was not optimal.")
            return None

        # Update the LP solution for the next iteration.
        x_lp = lp_result['solution']

        # Check the distance between the new LP solution and the rounded point.
        distance: float = 0
        for var_name in problem.integer_variable_names:
            val: float = x_lp[var_name]
            if abs(val - round(val)) > 1e-6:
                distance += abs(val - x_int[var_name])

        logger.debug(f"Pump {pump_iteration}: L1 distance = {distance:.4f}")

        # If the distance is zero, x_lp is integer-feasible. Success!
        if distance < 1e-6:
            logger.info(f"Feasibility Pump successful after {pump_iteration} pumps.")
            return {v_name: round(v_val) for v_name, v_val in x_lp.items()}

    logger.warning("Feasibility Pump exceeded max iterations.")
    return None


def _coefficient_diving(problem: MIPProblem,
                        initial_lp_solution: Dict[str, float],
                        initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Implements a Coefficient Diving Heuristic. 🤿

    This is a smarter version of the standard diving heuristic. Instead of just picking
    the fractional variable closest to an integer, it prioritizes fixing variables that
    appear in the most constraints (i.e., have the highest "lock count"). The intuition is
    that fixing these highly influential variables first will more quickly resolve the
    problem's structure and lead to a feasible solution.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        initial_lp_solution (Dict[str, float]): The starting LP solution.
        initial_constraints (List[Tuple[str, str, float]]): The starting constraints.

    Returns:
        Optional[Dict[str, float]]: A feasible integer solution if found, otherwise None.
    """
    logger.info("Attempting to find solution with Coefficient Diving Heuristic...")

    # Pre-calculate the "lock count" for each variable.
    lock_counts: Dict[str, int] = {v.VarName: 0 for v in problem.model.getVars()}
    for constr in problem.model.getConstrs():
        for i in range(problem.model.getRow(constr).size()):
            var_name: str = problem.model.getRow(constr).getVar(i).VarName
            lock_counts[var_name] += 1

    current_solution: Dict[str, float] = initial_lp_solution.copy()
    current_constraints: List[Tuple[str, str, float]] = initial_constraints.copy()

    for dive_iteration in range(problem.model.NumIntVars):
        # Find all fractional variables.
        fractional_vars: List[str] = []
        for var_name in problem.integer_variable_names:
            if var_name in current_solution and abs(current_solution[var_name] - round(current_solution[var_name])) > 1e-6:
                fractional_vars.append(var_name)

        if not fractional_vars:
            logger.info(f"Coefficient Diving successful after {dive_iteration} dives.")
            return {v_name: round(v_val) for v_name, v_val in current_solution.items()}

        # Select the fractional variable with the highest lock count to fix next.
        best_var_to_fix: str = max(fractional_vars, key=lambda vn: lock_counts.get(vn, 0))

        val_to_fix: float = current_solution[best_var_to_fix]
        rounded_val: float = round(val_to_fix)

        logger.debug(f"Coef. Dive {dive_iteration}: Fixing '{best_var_to_fix}' (lock count: {lock_counts.get(best_var_to_fix, 0)}) to {rounded_val}")
        
        # Add the fixing constraint and re-solve.
        current_constraints.append((best_var_to_fix, '==', float(rounded_val)))
        lp_result: Dict[str, Any] = solve_lp_relaxation(problem, current_constraints)

        if lp_result['status'] == 'OPTIMAL':
            current_solution = lp_result['solution']
        else:
            logger.info(f"Coefficient Diving failed at dive {dive_iteration}: subproblem became {lp_result['status']}.")
            return None

    logger.warning("Coefficient Diving heuristic exceeded max iterations.")
    return None


def find_initial_solution(problem: MIPProblem,
                          initial_lp_solution: Dict[str, float],
                          initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Runs a sequence of primal heuristics to find an initial integer-feasible solution.
    These are called "primal" because their goal is to find a feasible solution quickly,
    often at the start of the B&B process.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        initial_lp_solution (Dict[str, float]): The root node's LP solution.
        initial_constraints (List[Tuple[str, str, float]]): The root node's constraints.

    Returns:
        Optional[Dict[str, float]]: An integer solution if one is found, otherwise None.
    """
    # Try a simple heuristic first.
    solution: Optional[Dict[str, float]] = _diving_heuristic(problem, initial_lp_solution, initial_constraints)
    if solution:
        return solution

    # If the first one fails, try a more complex one.
    logger.info("Diving heuristic failed. Trying Feasibility Pump...")
    solution = _feasibility_pump(problem, initial_lp_solution)
    if solution:
        return solution

    return None


def run_periodic_heuristics(problem: MIPProblem,
                              current_node_solution: Dict[str, float],
                              incumbent_solution: Optional[Dict[str, float]],
                              config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Master function for running periodic and improvement heuristics. 🛠️

    This function orchestrates heuristics that are called periodically during the main
    B&B search. Unlike primal heuristics, their goal is to take an existing good
    solution (the incumbent) and find an even better one.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        current_node_solution (Dict[str, float]): The LP solution of the current B&B node.
        incumbent_solution (Optional[Dict[str, float]]): The best integer solution found so far.
        config (Dict[str, Any]): The solver configuration dictionary.

    Returns:
        Optional[Dict[str, Any]]: A full solution dictionary (with objective value) if a
        new, improved solution is found.
    """
    logger.info("--- Running Periodic/Improvement Heuristics ---")

    # --- RINS HEURISTIC ---
    # RINS is an improvement heuristic, so it requires an incumbent to work.
    if incumbent_solution:
        rins_solution: Optional[Dict[str, Any]] = _rins_heuristic(
            problem=problem,
            incumbent_solution=incumbent_solution,
            current_lp_solution=current_node_solution,
            time_limit=config.get('rins_time_limit', 5.0)
        )
        # The result from RINS is already a full solution dictionary.
        if rins_solution:
            return rins_solution
        
    return None