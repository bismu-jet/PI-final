import gurobipy as gp
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
    model.update() # Ensure model is up-to-date before we start
    
    constrs_to_remove_indices = []
    
    for i, constr in enumerate(model.getConstrs()):
        # A constraint is a singleton if it involves only one variable
        if model.getRow(constr).size() != 1:
            continue

        row = model.getRow(constr)
        var = row.getVar(0)
        coeff = row.getCoeff(0)
        rhs = constr.RHS
        sense = constr.Sense
        
        if abs(coeff) < 1e-9: continue # Should not happen, but a safe check

        # This constraint's information will be absorbed into the variable's bounds,
        # so we can mark it for removal.
        constrs_to_remove_indices.append(i)

        implied_val = rhs / coeff
        
        try:
            if sense == GRB.LESS_EQUAL:
                if coeff > 0: # e.g., 2x <= 10  =>  x <= 5
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")
                else: # e.g., -2x <= 10  =>  x >= -5
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
            
            elif sense == GRB.GREATER_EQUAL:
                if coeff > 0: # e.g., 2x >= 10  =>  x >= 5
                    if implied_val > var.LB:
                        var.LB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened LB of '{var.VarName}' to {implied_val}")
                else: # e.g., -2x >= 10  =>  x <= -5
                    if implied_val < var.UB:
                        var.UB = implied_val
                        logger.info(f"Constraint '{constr.ConstrName}' tightened UB of '{var.VarName}' to {implied_val}")

            elif sense == GRB.EQUAL: # e.g., 2x == 10  =>  x == 5
                # This constraint fixes the variable. We set both bounds to the same value.
                var.LB = implied_val
                var.UB = implied_val
                logger.info(f"Constraint '{constr.ConstrName}' fixed '{var.VarName}' to {implied_val}")
        
        except gp.GurobiError as e:
            # This error occurs if we try to set an invalid bound (e.g., UB < LB),
            # which proves the model is infeasible.
            logger.warning(f"Infeasibility detected by presolve while processing '{constr.ConstrName}'. Error: {e}")
            # For now, we will stop presolving and let the main solver find the infeasibility.
            # A more advanced implementation would set a status flag on the problem object.
            return

    if constrs_to_remove_indices:
        logger.info(f"Removing {len(constrs_to_remove_indices)} singleton constraints absorbed into variable bounds.")
        problem.remove_constraints_by_index(constrs_to_remove_indices)
        model.update()

def eliminate_redundant_constraints(problem: MIPProblem):
    """
    Identifies and removes redundant (dominated) constraints from the MIP problem.
    A constraint is dominated if another constraint with the same variables and
    coefficients is stricter.
    """
    logger.info("Starting presolve technique: Redundant Constraint Elimination...")
    model = problem.model
    constraints = model.getConstrs()
    
    # Group constraints by their structure (ignoring RHS and sense)
    constr_map: Dict[tuple, list] = {}
    for i, constr in enumerate(constraints):
        row = model.getRow(constr)
        # Create a canonical representation of the constraint's LHS
        # This tuple will be the key in our map
        vars_and_coeffs = tuple(sorted((row.getVar(j).VarName, row.getCoeff(j)) for j in range(row.size())))
        
        if vars_and_coeffs not in constr_map:
            constr_map[vars_and_coeffs] = []
        constr_map[vars_and_coeffs].append({'index': i, 'sense': constr.Sense, 'rhs': constr.RHS})

    constrs_to_remove_indices = []
    # Identify dominated constraints within each group
    for lhs, group in constr_map.items():
        if len(group) < 2:
            continue

        # Sub-group by sense (<= or >=)
        sense_groups: Dict[str, list] = {}
        for item in group:
            if item['sense'] not in sense_groups:
                sense_groups[item['sense']] = []
            sense_groups[item['sense']].append(item)

        # For '<=' constraints, keep only the one with the minimum RHS
        if GRB.LESS_EQUAL in sense_groups and len(sense_groups[GRB.LESS_EQUAL]) > 1:
            le_group = sense_groups[GRB.LESS_EQUAL]
            min_rhs_item = min(le_group, key=lambda x: x['rhs'])
            for item in le_group:
                if item['index'] != min_rhs_item['index']:
                    constrs_to_remove_indices.append(item['index'])

        # For '>=' constraints, keep only the one with the maximum RHS
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
    Performs coefficient tightening on constraints. This version includes safety
    checks to ensure it does not create invalid coefficients when variables
    have infinite bounds.
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
            can_tighten = True # Assume we can tighten unless we find an unbounded variable
            for j in range(row.size()):
                if j == k: continue
                var_j = row.getVar(j)
                coeff_j = row.getCoeff(j)
                
                # --- THIS IS THE FIX ---
                # Check for infinite bounds before calculating activity.
                # If a bound is infinite, we cannot safely tighten and must skip.
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or \
                   (coeff_j < 0 and var_j.UB == GRB.INFINITY):
                    can_tighten = False
                    break # Exit the inner loop, we cannot tighten this var_k
                
                if coeff_j > 0:
                    min_activity_rest += coeff_j * var_j.LB
                else:
                    min_activity_rest += coeff_j * var_j.UB
            
            if not can_tighten:
                continue # Move to the next variable in the constraint

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
            model.addConstr(c['expr'], sense=c['sense'], rhs=c['rhs'], name=c['name'])
        model.update()
#NEXT STEPS: PROBING HERE

def presolve(problem: MIPProblem):
    """
    The main presolve routine that calls all individual presolve techniques.
    
    Args:
        problem (MIPProblem): The MIPProblem to be presolved.
    """
    logger.info("--- Starting Presolve Pass ---")
    eliminate_redundant_constraints(problem)
    fix_variables_from_singletons(problem)
    tighten_coefficients(problem)
    logger.info("--- Presolve Pass Finished ---")