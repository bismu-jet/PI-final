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

# --- NEW FEASIBILITY PUMP HEURISTIC ---
def _feasibility_pump(problem: MIPProblem, 
                      initial_lp_solution: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Tries to find an integer-feasible solution using a Feasibility Pump heuristic.
    """
    logger.info("Attempting to find solution with Feasibility Pump...")
    
    x_lp = initial_lp_solution.copy()
    
    # Limit the number of pumping iterations
    for pump_iteration in range(20): # A common limit for this heuristic
        
        # 1. Round the current LP solution to get an integer point (x_int)
        x_int = {var_name: round(val) for var_name, val in x_lp.items()}
        
        # 2. Check if this integer point is feasible by chance
        # This requires a function to check constraints, which we will approximate for now.
        # A full implementation would check all original constraints.
        # For now, we rely on the distance minimization.
        
        # 3. Define a new objective: minimize the L1 distance to x_int
        objective_expr = gp.LinExpr()
        for var_name in problem.integer_variable_names:
            var = problem.model.getVarByName(var_name)
            if x_int[var_name] > 0.5: # If rounded value is 1
                objective_expr.add(var, -1)
            else: # If rounded value is 0
                objective_expr.add(var, 1)

        # 4. Solve the LP with this new "distance" objective
        lp_result = solve_lp_with_custom_objective(problem, objective_expr)
        
        if lp_result['status'] != 'OPTIMAL':
            logger.info(f"Feasibility Pump failed at iteration {pump_iteration}: distance LP was not optimal.")
            return None
            
        # 5. Update x_lp with the new solution
        x_lp = lp_result['solution']
        
        # 6. Check for convergence: are the integer variables now integer?
        is_integer = True
        distance = 0
        for var_name in problem.integer_variable_names:
            val = x_lp[var_name]
            distance += abs(val - x_int[var_name])
            if abs(val - round(val)) > 1e-6:
                is_integer = False
        
        logger.debug(f"Pump {pump_iteration}: L1 distance = {distance:.4f}")

        if is_integer:
            logger.info(f"Feasibility Pump successful after {pump_iteration} pumps.")
            return {v_name: round(v_val) for v_name, v_val in x_lp.items()}
            
    logger.warning("Feasibility Pump exceeded max iterations.")
    return None

# --- MODIFIED: Orchestrator function ---
def find_initial_solution(problem: MIPProblem, 
                          initial_lp_solution: Dict[str, float], 
                          initial_constraints: List[Tuple[str, str, float]]) -> Optional[Dict[str, float]]:
    """
    Runs a sequence of primal heuristics to find an initial integer-feasible solution.
    """
    
    # First, try the Diving Heuristic
    #solution = _diving_heuristic(problem, initial_lp_solution, initial_constraints)
    #if solution:
    #    return solution
        
    # If Diving fails, try the Feasibility Pump
    logger.info("Diving heuristic failed. Trying Feasibility Pump...")
    solution = _feasibility_pump(problem, initial_lp_solution)
    if solution:
        return solution
        
    return None
