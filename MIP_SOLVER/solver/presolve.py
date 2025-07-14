import gurobipy as gp
import math
from gurobipy import GRB
from typing import List, Dict
from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()


def fix_variables_from_singletons(problem: MIPProblem):
    """
    Finds constraints with only one variable (singletons) and uses them to
    tighten the variable's bounds. The now-redundant constraint is then removed.
    """
    logger.info("Starting presolve technique: Variable Fixing from Singleton Constraints...")
    model = problem.model
    model.update() 
    
    constrs_to_remove_indices = []
    
    # We iterate over a copy of the constraints list as we might modify the model
    for i, constr in enumerate(model.getConstrs()):
        if model.getRow(constr).size() != 1:
            continue

        row = model.getRow(constr)
        var = row.getVar(0)
        coeff = row.getCoeff(0)
        rhs = constr.RHS
        sense = constr.Sense
        
        if abs(coeff) < 1e-9: continue

        constrs_to_remove_indices.append(i)

        implied_val = rhs / coeff
        
        try:
            if sense == GRB.LESS_EQUAL:
                if coeff > 0:
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")
                else: 
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
            
            elif sense == GRB.GREATER_EQUAL:
                if coeff > 0:
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
                else:
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")

            elif sense == GRB.EQUAL:
                var.LB = implied_val
                var.UB = implied_val
                logger.info(f"Constraint '{constr.ConstrName}' fixed '{var.VarName}' to {implied_val}")
        
        except gp.GurobiError as e:
            logger.warning(f"Infeasibility detected by presolve while processing '{constr.ConstrName}'. Error: {e}")
            return

    if constrs_to_remove_indices:
        logger.info(f"Removing {len(constrs_to_remove_indices)} singleton constraints absorbed into variable bounds.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
        model.update()


def eliminate_redundant_constraints(problem: MIPProblem):
    """
    Identifies and removes redundant (dominated) constraints from the MIP problem.
    """
    logger.info("Starting presolve technique: Redundant Constraint Elimination...")
    model = problem.model
    constraints = model.getConstrs()
    
    constr_map: Dict[tuple, list] = {}
    for i, constr in enumerate(constraints):
        row = model.getRow(constr)
        vars_and_coeffs = tuple(sorted((row.getVar(j).VarName, row.getCoeff(j)) for j in range(row.size())))
        
        if vars_and_coeffs not in constr_map:
            constr_map[vars_and_coeffs] = []
        constr_map[vars_and_coeffs].append({'index': i, 'sense': constr.Sense, 'rhs': constr.RHS})

    constrs_to_remove_indices = []
    for lhs, group in constr_map.items():
        if len(group) < 2:
            continue

        sense_groups: Dict[str, list] = {}
        for item in group:
            if item['sense'] not in sense_groups:
                sense_groups[item['sense']] = []
            sense_groups[item['sense']].append(item)

        if GRB.LESS_EQUAL in sense_groups and len(sense_groups[GRB.LESS_EQUAL]) > 1:
            le_group = sense_groups[GRB.LESS_EQUAL]
            min_rhs_item = min(le_group, key=lambda x: x['rhs'])
            for item in le_group:
                if item['index'] != min_rhs_item['index']:
                    constrs_to_remove_indices.append(item['index'])

        if GRB.GREATER_EQUAL in sense_groups and len(sense_groups[GRB.GREATER_EQUAL]) > 1:
            ge_group = sense_groups[GRB.GREATER_EQUAL]
            max_rhs_item = max(ge_group, key=lambda x: x['rhs'])
            for item in ge_group:
                if item['index'] != max_rhs_item['index']:
                    constrs_to_remove_indices.append(item['index'])


    if constrs_to_remove_indices:
        logger.info(f"Found {len(constrs_to_remove_indices)} redundant constraints to remove.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
    else:
        logger.info("No redundant constraints found.")


def tighten_coefficients(problem: MIPProblem):
    """
    Performs coefficient tightening on constraints.
    """
    logger.info("Starting presolve technique: Coefficient Tightening...")
    model = problem.model
    model.update()

    constraints_to_add = []
    constrs_to_remove_indices = []

    for i, constr in enumerate(model.getConstrs()):
        if constr.Sense != GRB.LESS_EQUAL:
            continue

        row = model.getRow(constr)
        if row.size() < 2:
            continue

        modified = False
        new_coeffs = [row.getCoeff(j) for j in range(row.size())]
        new_vars = [row.getVar(j) for j in range(row.size())]

        for k in range(row.size()):
            var_k = row.getVar(k)
            if var_k.VType != GRB.BINARY or var_k.LB != 0 or var_k.UB != 1:
                continue
            
            coeff_k = row.getCoeff(k)
            if coeff_k <= 0:
                continue

            min_activity_rest = 0
            can_tighten = True
            for j in range(row.size()):
                if j == k: continue
                var_j = row.getVar(j)
                coeff_j = row.getCoeff(j)
                
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or \
                   (coeff_j < 0 and var_j.UB == GRB.INFINITY):
                    can_tighten = False
                    break 
                
                if coeff_j > 0:
                    min_activity_rest += coeff_j * var_j.LB
                else:
                    min_activity_rest += coeff_j * var_j.UB
            
            if not can_tighten:
                continue

            new_coeff_k = constr.RHS - min_activity_rest
            if new_coeff_k < coeff_k:
                logger.info(f"Tightening coefficient for var '{var_k.VarName}' in constr '{constr.ConstrName}' from {coeff_k} to {new_coeff_k:.4f}")
                new_coeffs[k] = new_coeff_k
                modified = True

        if modified:
            constrs_to_remove_indices.append(i)
            new_expr = gp.LinExpr(new_coeffs, new_vars)
            constraints_to_add.append({'name': constr.ConstrName + "_tightened", 'expr': new_expr, 'sense': GRB.LESS_EQUAL, 'rhs': constr.RHS})

    if constrs_to_remove_indices:
        logger.info(f"Replacing {len(constrs_to_remove_indices)} constraints with tightened versions.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
        model.update()
        for c in constraints_to_add:
            model.addConstr(c['expr'] <= c['rhs'], name=c['name'])
        model.update()


def propagate_bounds(problem: MIPProblem) -> int:
    """
    --- CORRECTED VERSION with ENHANCED LOGGING ---
    Iteratively tightens the bounds of variables based on the constraints.

    Returns:
        The number of tightenings, or -1 if infeasibility is detected.
    """
    logger.info("Starting presolve technique: Bound Propagation...")
    model, tightenings, TOLERANCE = problem.model, 0, 1e-9
    model.update()

    for constr in model.getConstrs():
        row = model.getRow(constr)
        if row.size() < 2: continue
        
        rhs, sense = constr.RHS, constr.Sense
        
        for i in range(row.size()):
            var_i = row.getVar(i)
            if var_i.LB > var_i.UB - TOLERANCE: continue

            coeff_i = row.getCoeff(i)
            if abs(coeff_i) < TOLERANCE: continue

            activity_rest_min, activity_rest_max = 0.0, 0.0
            for j in range(row.size()):
                if i == j: continue
                var_j, coeff_j = row.getVar(j), row.getCoeff(j)
                
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or (coeff_j < 0 and var_j.UB == GRB.INFINITY):
                    activity_rest_min = -GRB.INFINITY
                if (coeff_j > 0 and var_j.UB == GRB.INFINITY) or (coeff_j < 0 and var_j.LB == -GRB.INFINITY):
                    activity_rest_max = GRB.INFINITY

                if activity_rest_min != -GRB.INFINITY:
                    activity_rest_min += coeff_j * var_j.LB if coeff_j > 0 else coeff_j * var_j.UB
                if activity_rest_max != GRB.INFINITY:
                    activity_rest_max += coeff_j * var_j.UB if coeff_j > 0 else coeff_j * var_j.LB
            
            old_lb, old_ub = var_i.LB, var_i.UB
            
            # --- TIGHTEN UPPER BOUND ---
            new_ub_val = old_ub
            if coeff_i > 0 and sense in [GRB.LESS_EQUAL, GRB.EQUAL] and activity_rest_min != -GRB.INFINITY:
                new_ub_val = (rhs - activity_rest_min) / coeff_i
            elif coeff_i < 0 and sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and activity_rest_max != GRB.INFINITY:
                new_ub_val = (rhs - activity_rest_max) / coeff_i
            
            if abs(new_ub_val) < GRB.INFINITY and new_ub_val < old_ub - TOLERANCE:
                final_ub = math.floor(new_ub_val + TOLERANCE) if var_i.VType in [GRB.BINARY, GRB.INTEGER] else new_ub_val
                if final_ub < var_i.UB:
                    logger.debug(f"  [Bound Prop] UB of '{var_i.VarName}' tightened from {var_i.UB:.4f} to {final_ub:.4f} by constr '{constr.ConstrName}'")
                    var_i.UB = final_ub
                    tightenings += 1

            # --- TIGHTEN LOWER BOUND ---
            new_lb_val = old_lb
            if coeff_i > 0 and sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and activity_rest_max != -GRB.INFINITY:
                new_lb_val = (rhs - activity_rest_max) / coeff_i
            elif coeff_i < 0 and sense in [GRB.LESS_EQUAL, GRB.EQUAL] and activity_rest_min != -GRB.INFINITY:
                new_lb_val = (rhs - activity_rest_min) / coeff_i
            
            if abs(new_lb_val) < GRB.INFINITY and new_lb_val > old_lb + TOLERANCE:
                final_lb = math.ceil(new_lb_val - TOLERANCE) if var_i.VType in [GRB.BINARY, GRB.INTEGER] else new_lb_val
                if final_lb > var_i.LB:
                    logger.debug(f"  [Bound Prop] LB of '{var_i.VarName}' tightened from {var_i.LB:.4f} to {final_lb:.4f} by constr '{constr.ConstrName}'")
                    var_i.LB = final_lb
                    tightenings += 1

            # --- FINAL SAFETY CHECK ---
            if var_i.LB > var_i.UB + TOLERANCE:
                logger.warning(f"Presolve detected infeasibility: LB ({var_i.LB:.4f}) > UB ({var_i.UB:.4f}) for var '{var_i.VarName}' in constr '{constr.ConstrName}'")
                return -1

    if tightenings > 0:
        logger.info(f"Bound propagation pass finished. Total tightenings: {tightenings}.")
        model.update()
    else:
        logger.info("Bound propagation pass finished. No new tightenings found.")
        
    return tightenings


def probe_binary_variables(problem: MIPProblem):
    """
    --- REVISED: Performs probing on binary variables more efficiently and with detailed logging. ---
    
    For each binary variable, it checks if fixing it to 0 or 1 leads to
    an infeasible subproblem, allowing the variable to be fixed to the other value.
    This version uses a single model copy for better performance.
    """
    logger.info("Starting presolve technique: Probing...")
    model = problem.model
    model.update()

    binary_vars = [v for v in model.getVars() if v.VType == GRB.BINARY and v.LB != v.UB]
    if not binary_vars:
        logger.info("Probing: No unfixed binary variables to probe.")
        return

    vars_to_fix_to_0 = []
    vars_to_fix_to_1 = []

    probe_model = None
    try:
        probe_model = model.copy()
        probe_model.setParam('OutputFlag', 0)
        probe_model.setParam('TimeLimit', 1) 
        probe_model.setParam('Method', 0) 

        total_probes = len(binary_vars)
        logger.info(f"Probing {total_probes} binary variables...")

        for i, var in enumerate(binary_vars):
            p_var = probe_model.getVarByName(var.VarName)
            logger.debug(f"Probing [{i+1}/{total_probes}]: '{var.VarName}'")
            
            # --- Probe by fixing to 1 (to see if we can fix to 0) ---
            original_lb = p_var.LB
            p_var.LB = 1.0
            probe_model.optimize()
            if probe_model.Status == GRB.INFEASIBLE:
                vars_to_fix_to_0.append(var.VarName)
                logger.debug(f"  -> Probe result for '{var.VarName}'=1 is INFEASIBLE. Can fix to 0.")
            p_var.LB = original_lb # Restore bound

            # --- Probe by fixing to 0 (to see if we can fix to 1) ---
            # We only do this if the first probe didn't already find a fixing
            if var.VarName not in vars_to_fix_to_0:
                original_ub = p_var.UB
                p_var.UB = 0.0
                probe_model.optimize()
                if probe_model.Status == GRB.INFEASIBLE:
                    vars_to_fix_to_1.append(var.VarName)
                    logger.debug(f"  -> Probe result for '{var.VarName}'=0 is INFEASIBLE. Can fix to 1.")
                p_var.UB = original_ub # Restore bound
            
    finally:
        if probe_model:
            probe_model.dispose()

    # Apply the deductions to the original model
    if vars_to_fix_to_0:
        logger.info(f"Probing fixed {len(vars_to_fix_to_0)} variables to 0.")
        for var_name in vars_to_fix_to_0:
            model.getVarByName(var_name).UB = 0.0
            
    if vars_to_fix_to_1:
        logger.info(f"Probing fixed {len(vars_to_fix_to_1)} variables to 1.")
        for var_name in vars_to_fix_to_1:
            model.getVarByName(var_name).LB = 1.0
    
    total_fixed = len(vars_to_fix_to_0) + len(vars_to_fix_to_1)
    if total_fixed > 0:
        logger.info(f"Probing Summary: Fixed a total of {total_fixed} variables.")
        model.update()
    else:
        logger.info("Probing did not find any variable fixings.")

def presolve(problem: MIPProblem):
    """
    --- REVISED: The main presolve routine. --- 
    
    Calls presolve techniques iteratively and halts if infeasibility is proven.
    """
    logger.info("--- Starting Presolve Phase ---")
    
    max_passes = 4
    for i in range(max_passes):
        logger.info(f"Presolve Iteration {i+1}/{max_passes}...")
        
        fix_variables_from_singletons(problem)
        eliminate_redundant_constraints(problem)
        tighten_coefficients(problem)
        
        changes_found = propagate_bounds(problem)
        
        # --- Handle Infeasibility Signal ---
        if changes_found == -1:
            logger.error("Model has been proven infeasible during bound propagation. Halting solver.")
            # Optionally set a status on the problem object to be used by the caller
            # problem.model.setAttr("Status", GRB.INFEASIBLE) 
            return # Exit presolve entirely

        if changes_found == 0:
            logger.info("Presolve pass completed with no new bound changes. Exiting loop.")
            break
            
    # Probing should only run if the model is still potentially feasible
    probe_binary_variables(problem)
    
    logger.info("--- Presolve Phase Finished ---")