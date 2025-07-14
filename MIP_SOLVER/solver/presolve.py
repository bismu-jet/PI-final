import gurobipy as gp
import math
from gurobipy import GRB
from typing import List, Dict, Any, Tuple

# --- Importing custom modules for the solver ---
from solver.problem import MIPProblem # To load and manage the MIP problem data.
from solver.utilities import setup_logger # Utility for logging solver progress.

# --- Setup the logger ---
# Creates a logger object to print progress and debug information for the presolve phase.
logger = setup_logger()


def fix_variables_from_singletons(problem: MIPProblem) -> None:
    """
    Finds constraints with only one variable (singletons) and uses them to
    tighten the variable's bounds. The now-redundant constraint is then removed.
    For example, a constraint '3x <= 9' implies 'x <= 3'. This new upper bound for x
    is applied, and the original constraint is removed.
    
    Args:
        problem (MIPProblem): The MIP problem instance to be modified.
    """
    logger.info("Starting presolve technique: Variable Fixing from Singleton Constraints...")
    model: gp.Model = problem.model
    # The model must be updated to ensure all recent changes are reflected.
    model.update() 
    
    # Store indices of constraints that can be removed.
    constrs_to_remove_indices: List[int] = []
    
    # Iterate over all constraints to find singletons.
    for i, constr in enumerate(model.getConstrs()):
        # Check if the constraint involves only one variable.
        if model.getRow(constr).size() != 1:
            continue

        # Extract the variable, its coefficient, and constraint details.
        row: gp.LinExpr = model.getRow(constr)
        var: gp.Var = row.getVar(0)
        coeff: float = row.getCoeff(0)
        rhs: float = constr.RHS
        sense: str = constr.Sense
        
        # Skip if the coefficient is effectively zero.
        if abs(coeff) < 1e-9: continue

        # This constraint is a singleton and will be removed after its information is absorbed.
        constrs_to_remove_indices.append(i)

        # Calculate the value implied by the constraint.
        implied_val: float = rhs / coeff
        
        try:
            # --- Update variable bounds based on the constraint sense ---
            if sense == GRB.LESS_EQUAL: # e.g., a*x <= b
                if coeff > 0: # a > 0  => x <= b/a (new upper bound)
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")
                else: # a < 0 => x >= b/a (new lower bound)
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
            
            elif sense == GRB.GREATER_EQUAL: # e.g., a*x >= b
                if coeff > 0: # a > 0 => x >= b/a (new lower bound)
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
                else: # a < 0 => x <= b/a (new upper bound)
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")

            elif sense == GRB.EQUAL: # e.g., a*x = b
                # The variable can be fixed to a single value.
                var.LB = implied_val
                var.UB = implied_val
                logger.info(f"Constraint '{constr.ConstrName}' fixed '{var.VarName}' to {implied_val}")
        
        except gp.GurobiError as e:
            # This error occurs if a new bound contradicts an existing one (e.g., new UB < current LB).
            logger.warning(f"Infeasibility detected by presolve while processing '{constr.ConstrName}'. Error: {e}")
            return

    # Remove all the now-redundant singleton constraints from the model.
    if constrs_to_remove_indices:
        logger.info(f"Removing {len(constrs_to_remove_indices)} singleton constraints absorbed into variable bounds.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
        model.update()


def eliminate_redundant_constraints(problem: MIPProblem) -> None:
    """
    Identifies and removes redundant (dominated) constraints from the MIP problem.
    For example, if we have two constraints `x+y <= 10` and `x+y <= 5`, the first
    one is redundant because any solution satisfying the second one will always
    satisfy the first.
    
    Args:
        problem (MIPProblem): The MIP problem instance to be modified.
    """
    logger.info("Starting presolve technique: Redundant Constraint Elimination...")
    model: gp.Model = problem.model
    constraints: List[gp.Constr] = model.getConstrs()
    
    # A map to group constraints by their variables and coefficients.
    constr_map: Dict[tuple, list] = {}
    for i, constr in enumerate(constraints):
        row: gp.LinExpr = model.getRow(constr)
        # Create a unique, sorted signature for the left-hand side (LHS) of the constraint.
        vars_and_coeffs: Tuple = tuple(sorted((row.getVar(j).VarName, row.getCoeff(j)) for j in range(row.size())))
        
        # Group constraints with the exact same LHS.
        if vars_and_coeffs not in constr_map:
            constr_map[vars_and_coeffs] = []
        constr_map[vars_and_coeffs].append({'index': i, 'sense': constr.Sense, 'rhs': constr.RHS})

    constrs_to_remove_indices: List[int] = []
    # Iterate through groups of constraints with the same LHS.
    for lhs, group in constr_map.items():
        if len(group) < 2:
            continue

        # Further group by constraint sense (<=, >=, ==).
        sense_groups: Dict[str, list] = {}
        for item in group:
            if item['sense'] not in sense_groups:
                sense_groups[item['sense']] = []
            sense_groups[item['sense']].append(item)

        # For '<=' constraints, only the one with the smallest RHS is necessary.
        if GRB.LESS_EQUAL in sense_groups and len(sense_groups[GRB.LESS_EQUAL]) > 1:
            le_group: List[Dict] = sense_groups[GRB.LESS_EQUAL]
            # Find the most restrictive constraint (minimum RHS).
            min_rhs_item: Dict = min(le_group, key=lambda x: x['rhs'])
            # Mark all others in this group for removal.
            for item in le_group:
                if item['index'] != min_rhs_item['index']:
                    constrs_to_remove_indices.append(item['index'])

        # For '>=' constraints, only the one with the largest RHS is necessary.
        if GRB.GREATER_EQUAL in sense_groups and len(sense_groups[GRB.GREATER_EQUAL]) > 1:
            ge_group: List[Dict] = sense_groups[GRB.GREATER_EQUAL]
            # Find the most restrictive constraint (maximum RHS).
            max_rhs_item: Dict = max(ge_group, key=lambda x: x['rhs'])
            # Mark all others in this group for removal.
            for item in ge_group:
                if item['index'] != max_rhs_item['index']:
                    constrs_to_remove_indices.append(item['index'])

    # Remove all identified redundant constraints.
    if constrs_to_remove_indices:
        logger.info(f"Found {len(constrs_to_remove_indices)} redundant constraints to remove.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
    else:
        logger.info("No redundant constraints found.")


def tighten_coefficients(problem: MIPProblem) -> None:
    """
    Performs coefficient tightening on constraints. For a binary variable in a constraint,
    this technique can sometimes reduce its coefficient, making the LP relaxation tighter.
    This is especially useful for generating stronger cuts later.
    
    Args:
        problem (MIPProblem): The MIP problem instance to be modified.
    """
    logger.info("Starting presolve technique: Coefficient Tightening...")
    model: gp.Model = problem.model
    model.update()

    constraints_to_add: List[Dict[str, Any]] = []
    constrs_to_remove_indices: List[int] = []

    # Iterate over all constraints.
    for i, constr in enumerate(model.getConstrs()):
        # This implementation focuses on '<=' constraints.
        if constr.Sense != GRB.LESS_EQUAL:
            continue

        row: gp.LinExpr = model.getRow(constr)
        if row.size() < 2:
            continue

        modified: bool = False
        new_coeffs: List[float] = [row.getCoeff(j) for j in range(row.size())]
        new_vars: List[gp.Var] = [row.getVar(j) for j in range(row.size())]

        # Try to tighten the coefficient for each variable in the constraint.
        for k in range(row.size()):
            var_k: gp.Var = row.getVar(k)
            # This technique applies to standard binary variables (0 or 1).
            if var_k.VType != GRB.BINARY or var_k.LB != 0 or var_k.UB != 1:
                continue
            
            coeff_k: float = row.getCoeff(k)
            # Only positive coefficients can be tightened this way.
            if coeff_k <= 0:
                continue

            # Calculate the minimum possible value of the rest of the constraint (LHS excluding var_k).
            min_activity_rest: float = 0
            can_tighten: bool = True
            for j in range(row.size()):
                if j == k: continue
                var_j: gp.Var = row.getVar(j)
                coeff_j: float = row.getCoeff(j)
                
                # Cannot calculate a finite minimum activity if any variable is unbounded.
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or \
                   (coeff_j < 0 and var_j.UB == GRB.INFINITY):
                    can_tighten = False
                    break 
                
                # To find the minimum activity, use the lower bound for positive coefficients
                # and the upper bound for negative coefficients.
                if coeff_j > 0:
                    min_activity_rest += coeff_j * var_j.LB
                else:
                    min_activity_rest += coeff_j * var_j.UB
            
            if not can_tighten:
                continue

            # The new, potentially tighter coefficient is derived from the RHS and min_activity.
            new_coeff_k: float = constr.RHS - min_activity_rest
            if new_coeff_k < coeff_k:
                logger.info(f"Tightening coefficient for var '{var_k.VarName}' in constr '{constr.ConstrName}' from {coeff_k} to {new_coeff_k:.4f}")
                new_coeffs[k] = new_coeff_k
                modified = True

        # If any coefficient was tightened, the original constraint must be replaced.
        if modified:
            constrs_to_remove_indices.append(i)
            new_expr: gp.LinExpr = gp.LinExpr(new_coeffs, new_vars)
            constraints_to_add.append({'name': constr.ConstrName + "_tightened", 'expr': new_expr, 'sense': GRB.LESS_EQUAL, 'rhs': constr.RHS})

    # Atomically replace old constraints with their new tightened versions.
    if constrs_to_remove_indices:
        logger.info(f"Replacing {len(constrs_to_remove_indices)} constraints with tightened versions.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
        model.update()
        for c in constraints_to_add:
            model.addConstr(c['expr'] <= c['rhs'], name=c['name'])
        model.update()


def propagate_bounds(problem: MIPProblem) -> int:
    """
    Iteratively tightens the bounds of variables based on the constraints.
    For each variable in a constraint, it calculates the tightest possible
    bounds it could have given the bounds of all other variables in that constraint.
    
    Args:
        problem (MIPProblem): The MIP problem instance.

    Returns:
        int: The number of tightenings performed, or -1 if infeasibility is detected.
    """
    logger.info("Starting presolve technique: Bound Propagation...")
    model, tightenings, TOLERANCE = problem.model, 0, 1e-9
    model.update()

    # Loop through every constraint in the model.
    for constr in model.getConstrs():
        row: gp.LinExpr = model.getRow(constr)
        if row.size() < 2: continue # Singleton constraints are handled separately.
        
        rhs, sense = constr.RHS, constr.Sense
        
        # For each variable in the constraint, try to tighten its bounds.
        for i in range(row.size()):
            var_i: gp.Var = row.getVar(i)
            # Skip if the variable is already fixed.
            if var_i.LB > var_i.UB - TOLERANCE: continue

            coeff_i: float = row.getCoeff(i)
            if abs(coeff_i) < TOLERANCE: continue

            # Calculate the minimum and maximum possible values (activity) of the rest of the constraint.
            activity_rest_min, activity_rest_max = 0.0, 0.0
            for j in range(row.size()):
                if i == j: continue
                var_j, coeff_j = row.getVar(j), row.getCoeff(j)
                
                # Check for unbounded variables that make finite activity calculation impossible.
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or (coeff_j < 0 and var_j.UB == GRB.INFINITY):
                    activity_rest_min = -GRB.INFINITY
                if (coeff_j > 0 and var_j.UB == GRB.INFINITY) or (coeff_j < 0 and var_j.LB == -GRB.INFINITY):
                    activity_rest_max = GRB.INFINITY

                # Sum up the min/max activities based on variable bounds and coefficients.
                if activity_rest_min != -GRB.INFINITY:
                    activity_rest_min += coeff_j * var_j.LB if coeff_j > 0 else coeff_j * var_j.UB
                if activity_rest_max != GRB.INFINITY:
                    activity_rest_max += coeff_j * var_j.UB if coeff_j > 0 else coeff_j * var_j.LB
            
            old_lb, old_ub = var_i.LB, var_i.UB
            
            # --- Try to Tighten the Upper Bound (UB) of var_i ---
            new_ub_val: float = old_ub
            # Based on: coeff_i * var_i <= RHS - activity_rest_min
            if coeff_i > 0 and sense in [GRB.LESS_EQUAL, GRB.EQUAL] and activity_rest_min != -GRB.INFINITY:
                new_ub_val = (rhs - activity_rest_min) / coeff_i
            # Based on: coeff_i * var_i >= RHS - activity_rest_max (dividing by negative coeff_i flips inequality)
            elif coeff_i < 0 and sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and activity_rest_max != GRB.INFINITY:
                new_ub_val = (rhs - activity_rest_max) / coeff_i
            
            if abs(new_ub_val) < GRB.INFINITY and new_ub_val < old_ub - TOLERANCE:
                # For integer/binary variables, round down the new bound.
                final_ub = math.floor(new_ub_val + TOLERANCE) if var_i.VType in [GRB.BINARY, GRB.INTEGER] else new_ub_val
                if final_ub < var_i.UB:
                    logger.debug(f"  [Bound Prop] UB of '{var_i.VarName}' tightened from {var_i.UB:.4f} to {final_ub:.4f} by constr '{constr.ConstrName}'")
                    var_i.UB = final_ub
                    tightenings += 1

            # --- Try to Tighten the Lower Bound (LB) of var_i ---
            new_lb_val: float = old_lb
            # Based on: coeff_i * var_i >= RHS - activity_rest_max
            if coeff_i > 0 and sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and activity_rest_max != GRB.INFINITY:
                new_lb_val = (rhs - activity_rest_max) / coeff_i
            # Based on: coeff_i * var_i <= RHS - activity_rest_min (dividing by negative coeff_i flips inequality)
            elif coeff_i < 0 and sense in [GRB.LESS_EQUAL, GRB.EQUAL] and activity_rest_min != -GRB.INFINITY:
                new_lb_val = (rhs - activity_rest_min) / coeff_i
            
            if abs(new_lb_val) < GRB.INFINITY and new_lb_val > old_lb + TOLERANCE:
                # For integer/binary variables, round up the new bound.
                final_lb = math.ceil(new_lb_val - TOLERANCE) if var_i.VType in [GRB.BINARY, GRB.INTEGER] else new_lb_val
                if final_lb > var_i.LB:
                    logger.debug(f"  [Bound Prop] LB of '{var_i.VarName}' tightened from {var_i.LB:.4f} to {final_lb:.4f} by constr '{constr.ConstrName}'")
                    var_i.LB = final_lb
                    tightenings += 1

            # --- Check for Infeasibility ---
            # If at any point a variable's lower bound becomes greater than its upper bound, the model is infeasible.
            if var_i.LB > var_i.UB + TOLERANCE:
                logger.warning(f"Presolve detected infeasibility: LB ({var_i.LB:.4f}) > UB ({var_i.UB:.4f}) for var '{var_i.VarName}' in constr '{constr.ConstrName}'")
                return -1

    if tightenings > 0:
        logger.info(f"Bound propagation pass finished. Total tightenings: {tightenings}.")
        model.update()
    else:
        logger.info("Bound propagation pass finished. No new tightenings found.")
        
    return tightenings


def probe_binary_variables(problem: MIPProblem) -> None:
    """
    Performs probing on binary variables. For each binary variable, it temporarily
    fixes it to 0 and then to 1. If either temporary fix leads to an infeasible
    subproblem, the variable can be permanently fixed to the opposite value.
    This is a powerful but potentially time-consuming technique.
    
    Args:
        problem (MIPProblem): The MIP problem instance.
    """
    logger.info("Starting presolve technique: Probing...")
    model: gp.Model = problem.model
    model.update()

    # Get a list of binary variables that are not already fixed.
    binary_vars: List[gp.Var] = [v for v in model.getVars() if v.VType == GRB.BINARY and v.LB != v.UB]
    if not binary_vars:
        logger.info("Probing: No unfixed binary variables to probe.")
        return

    # Store variables that can be fixed based on probing results.
    vars_to_fix_to_0: List[str] = []
    vars_to_fix_to_1: List[str] = []

    probe_model: gp.Model = None
    try:
        # Create a copy of the model for probing to avoid modifying the original.
        probe_model = model.copy()
        # Configure the probe model for speed: disable output, set a time limit, and use a fast method (simplex).
        probe_model.setParam('OutputFlag', 0)
        probe_model.setParam('TimeLimit', 1) 
        probe_model.setParam('Method', 0) 

        total_probes: int = len(binary_vars)
        logger.info(f"Probing {total_probes} binary variables...")

        for i, var in enumerate(binary_vars):
            # Get the corresponding variable in the copied model.
            p_var: gp.Var = probe_model.getVarByName(var.VarName)
            logger.debug(f"Probing [{i+1}/{total_probes}]: '{var.VarName}'")
            
            # --- Probe 1: Try fixing var to 1 ---
            original_lb: float = p_var.LB
            p_var.LB = 1.0
            probe_model.optimize()
            # If fixing to 1 makes the model infeasible, we can permanently fix the original variable to 0.
            if probe_model.Status == GRB.INFEASIBLE:
                vars_to_fix_to_0.append(var.VarName)
                logger.debug(f"  -> Probe result for '{var.VarName}'=1 is INFEASIBLE. Can fix to 0.")
            p_var.LB = original_lb # Always restore the bound for the next probe.

            # --- Probe 2: Try fixing var to 0 ---
            # Only probe 0 if probing 1 didn't already yield a result.
            if var.VarName not in vars_to_fix_to_0:
                original_ub: float = p_var.UB
                p_var.UB = 0.0
                probe_model.optimize()
                # If fixing to 0 makes the model infeasible, we can permanently fix the original variable to 1.
                if probe_model.Status == GRB.INFEASIBLE:
                    vars_to_fix_to_1.append(var.VarName)
                    logger.debug(f"  -> Probe result for '{var.VarName}'=0 is INFEASIBLE. Can fix to 1.")
                p_var.UB = original_ub # Restore the bound.
            
    finally:
        # Important: Dispose of the Gurobi model copy to free memory.
        if probe_model:
            probe_model.dispose()

    # --- Apply all deductions found during probing to the original model ---
    if vars_to_fix_to_0:
        logger.info(f"Probing fixed {len(vars_to_fix_to_0)} variables to 0.")
        for var_name in vars_to_fix_to_0:
            model.getVarByName(var_name).UB = 0.0
            
    if vars_to_fix_to_1:
        logger.info(f"Probing fixed {len(vars_to_fix_to_1)} variables to 1.")
        for var_name in vars_to_fix_to_1:
            model.getVarByName(var_name).LB = 1.0
    
    total_fixed: int = len(vars_to_fix_to_0) + len(vars_to_fix_to_1)
    if total_fixed > 0:
        logger.info(f"Probing Summary: Fixed a total of {total_fixed} variables.")
        model.update()
    else:
        logger.info("Probing did not find any variable fixings.")


def presolve(problem: MIPProblem) -> None:
    """
    The main presolve routine. It calls various presolve techniques iteratively
    to simplify the MIP problem as much as possible before starting the main
    Branch and Bound solver.
    
    Args:
        problem (MIPProblem): The MIP problem instance to be presolved.
    """
    logger.info("--- Starting Presolve Phase ---")
    
    # Run the presolve techniques in a loop, as one reduction can enable others.
    max_passes: int = 1
    for i in range(max_passes):
        logger.info(f"Presolve Iteration {i+1}/{max_passes}...")
        
        # --- Call individual presolve techniques in a logical order ---
        fix_variables_from_singletons(problem)
        eliminate_redundant_constraints(problem)
        tighten_coefficients(problem)
        
        # Bound propagation is often a good technique to run repeatedly.
        changes_found: int = propagate_bounds(problem)
        
        # --- Handle Infeasibility ---
        # If any technique proves infeasibility, we can stop immediately.
        if changes_found == -1:
            logger.error("Model has been proven infeasible during bound propagation. Halting presolve.")
            # problem.set_status("INFEASIBLE") # Example of how to signal this to the solver
            return # Exit presolve entirely.

        # If a full pass makes no changes, further passes are unlikely to help.
        if changes_found == 0:
            logger.info("Presolve pass completed with no new bound changes. Exiting loop.")
            break
            
    # Probing is often run last as it can be more computationally expensive.
    probe_binary_variables(problem)
    
    logger.info("--- Presolve Phase Finished ---")