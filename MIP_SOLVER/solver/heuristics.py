import gurobipy as gp
from gurobipy import GRB
import math
from typing import List, Dict, Optional, Tuple, Any

from solver.problem import MIPProblem
from solver.gurobi_interface import solve_lp_relaxation, solve_lp_with_custom_objective
from solver.utilities import setup_logger

logger = setup_logger()

def _diving_heuristic(problem: MIPProblem,
                      initial_lp_solution: Dict[str, float],
                      initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Tries to find an integer-feasible solution using a Diving Heuristic.
    """
    logger.info("Attempting to find solution with Diving Heuristic...")

    current_solution = initial_lp_solution.copy()
    current_constraints = initial_constraints.copy()

    for dive_iteration in range(problem.model.NumIntVars * 2):
        fractional_vars = []
        for var_name in problem.integer_variable_names:
            if var_name in current_solution:
                val = current_solution[var_name]
                if abs(val - round(val)) > 1e-6:
                    distance = 0.5 - abs(val - math.floor(val) - 0.5)
                    fractional_vars.append((var_name, distance))

        if not fractional_vars:
            logger.info(f"Diving heuristic successful after {dive_iteration} dives.")
            return {v_name: round(v_val) for v_name, v_val in current_solution.items()}

        fractional_vars.sort(key=lambda x: x[1])
        var_to_fix, _ = fractional_vars[0]

        val_to_fix = current_solution[var_to_fix]
        rounded_val = round(val_to_fix)

        logger.debug(f"Dive {dive_iteration}: Fixing '{var_to_fix}' from {val_to_fix:.4f} to {rounded_val}")

        current_constraints.append((var_to_fix, '==', float(rounded_val)))

        lp_result = solve_lp_relaxation(problem, current_constraints)

        if lp_result['status'] == 'OPTIMAL':
            current_solution = lp_result['solution']
        else:
            logger.info(f"Diving heuristic failed at dive {dive_iteration}: subproblem became {lp_result['status']}.")
            return None

    logger.warning("Diving heuristic exceeded max iterations.")
    return None

def _feasibility_pump(problem: MIPProblem,
                      initial_lp_solution: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Tries to find an integer-feasible solution using a Feasibility Pump heuristic.
    This is a corrected, robust version.
    """
    logger.info("Attempting to find solution with Feasibility Pump...")

    x_lp = initial_lp_solution.copy()

    for pump_iteration in range(20):

        x_int = {var_name: round(val) for var_name, val in x_lp.items()}

        objective_coeffs = {}
        for var_name in problem.integer_variable_names:
            if x_int.get(var_name, 0) > 0.5: # If rounded value is 1
                objective_coeffs[var_name] = -1.0
            else: # If rounded value is 0
                objective_coeffs[var_name] = 1.0

        lp_result = solve_lp_with_custom_objective(problem, objective_coeffs)

        if lp_result['status'] != 'OPTIMAL':
            logger.info(f"Feasibility Pump failed at iteration {pump_iteration}: distance LP was not optimal.")
            return None

        x_lp = lp_result['solution']

        distance = 0
        for var_name in problem.integer_variable_names:
            val = x_lp[var_name]
            if abs(val - round(val)) > 1e-6:
                distance += abs(val - x_int[var_name])

        logger.debug(f"Pump {pump_iteration}: L1 distance = {distance:.4f}")

        if distance < 1e-6:
            logger.info(f"Feasibility Pump successful after {pump_iteration} pumps.")
            return {v_name: round(v_val) for v_name, v_val in x_lp.items()}

    logger.warning("Feasibility Pump exceeded max iterations.")
    return None

def _coefficient_diving(problem: MIPProblem,
                        initial_lp_solution: Dict[str, float],
                        initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    --- NEW: Coefficient Diving Heuristic ---
    A diving heuristic that intelligently selects variables to fix based on
    how "locked" they are by the constraints.
    """
    logger.info("Attempting to find solution with Coefficient Diving Heuristic...")

    # Pre-calculate lock counts for all variables
    lock_counts = {v.VarName: 0 for v in problem.model.getVars()}
    for constr in problem.model.getConstrs():
        for i in range(problem.model.getRow(constr).size()):
            var_name = problem.model.getRow(constr).getVar(i).VarName
            lock_counts[var_name] += 1

    current_solution = initial_lp_solution.copy()
    current_constraints = initial_constraints.copy()

    for dive_iteration in range(problem.model.NumIntVars):
        fractional_vars = []
        for var_name in problem.integer_variable_names:
            if var_name in current_solution and abs(current_solution[var_name] - round(current_solution[var_name])) > 1e-6:
                fractional_vars.append(var_name)

        if not fractional_vars:
            logger.info(f"Coefficient Diving successful after {dive_iteration} dives.")
            return {v_name: round(v_val) for v_name, v_val in current_solution.items()}

        # Select the fractional variable with the highest lock count
        best_var_to_fix = max(fractional_vars, key=lambda vn: lock_counts.get(vn, 0))
        
        val_to_fix = current_solution[best_var_to_fix]
        rounded_val = round(val_to_fix)
        
        logger.debug(f"Coef. Dive {dive_iteration}: Fixing '{best_var_to_fix}' (lock count: {lock_counts.get(best_var_to_fix, 0)}) to {rounded_val}")
        current_constraints.append((best_var_to_fix, '==', float(rounded_val)))
        
        lp_result = solve_lp_relaxation(problem, current_constraints)
        
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
    """
    solution = _diving_heuristic(problem, initial_lp_solution, initial_constraints)
    if solution:
        return solution

    logger.info("Diving heuristic failed. Trying Feasibility Pump...")
    solution = _feasibility_pump(problem, initial_lp_solution)
    if solution:
        return solution

    return None

def run_periodic_heuristics(problem: MIPProblem,
                           lp_solution: Dict[str, float],
                           local_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    --- NEW: Master function for periodic heuristics ---
    Runs more expensive heuristics that are not run at the root node.
    """
    logger.info("--- Running Periodic Heuristics ---")
    solution = _coefficient_diving(problem, lp_solution, local_constraints)
    if solution:
        # As before, we must verify the heuristic solution can be extended
        # to a fully feasible solution for all variables.
        fixed_vars_constraints = []
        for var_name, value in solution.items():
            if var_name in problem.integer_variable_names:
                 fixed_vars_constraints.append((var_name, '==', float(round(value))))
        
        completion_lp_result = solve_lp_relaxation(problem, fixed_vars_constraints)

        if completion_lp_result['status'] == 'OPTIMAL':
            logger.info("Coefficient Diving found and verified a new feasible solution.")
            return completion_lp_result['solution']
        else:
            logger.warning("Coefficient Diving solution was not extendable to a feasible solution.")

    return None
