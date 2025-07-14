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
    model.update() 
    
    constrs_to_remove_indices = []
    
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


def probe_binary_variables(problem: MIPProblem):
    """
    --- NEW: Performs probing on binary variables to find fixings. ---
    
    For each binary variable, it checks if fixing it to 0 or 1 leads to
    an infeasible subproblem, allowing the variable to be fixed to the other value.
    """
    logger.info("Starting presolve technique: Probing...")
    model = problem.model
    model.update()

    binary_vars = [v for v in model.getVars() if v.VType == GRB.BINARY]
    
    vars_to_fix_to_0 = []
    vars_to_fix_to_1 = []

    for var in binary_vars:
        logger.info("Probing...")
        # Skip variables that are already fixed
        if var.LB == var.UB:
            continue
            
        # --- Probe by fixing to 1 ---
        probe_model_1 = model.copy()
        probe_model_1.setParam('OutputFlag', 0)
        probe_model_1.setParam('TimeLimit', 1) # Short time limit for presolve
        p_var_1 = probe_model_1.getVarByName(var.VarName)
        p_var_1.LB = 1.0
        probe_model_1.optimize()
        if probe_model_1.Status == GRB.INFEASIBLE:
            vars_to_fix_to_0.append(var.VarName)
        probe_model_1.dispose()

        # --- Probe by fixing to 0 ---
        probe_model_0 = model.copy()
        probe_model_0.setParam('OutputFlag', 0)
        probe_model_0.setParam('TimeLimit', 1)
        p_var_0 = probe_model_0.getVarByName(var.VarName)
        p_var_0.UB = 0.0
        probe_model_0.optimize()
        if probe_model_0.Status == GRB.INFEASIBLE:
            vars_to_fix_to_1.append(var.VarName)
        probe_model_0.dispose()

    # Apply the deductions to the original model
    if vars_to_fix_to_0:
        logger.info(f"Probing fixed {len(vars_to_fix_to_0)} variables to 0.")
        for var_name in vars_to_fix_to_0:
            model.getVarByName(var_name).UB = 0.0
            
    if vars_to_fix_to_1:
        logger.info(f"Probing fixed {len(vars_to_fix_to_1)} variables to 1.")
        for var_name in vars_to_fix_to_1:
            model.getVarByName(var_name).LB = 1.0
    
    if vars_to_fix_to_0 or vars_to_fix_to_1:
        model.update()
    else:
        logger.info("Probing did not find any variable fixings.")


def presolve(problem: MIPProblem):
    """
    The main presolve routine that calls all individual presolve techniques.
    """
    logger.info("--- Starting Presolve Pass ---")
    eliminate_redundant_constraints(problem)
    fix_variables_from_singletons(problem)
    tighten_coefficients(problem)
    probe_binary_variables(problem)
    logger.info("--- Presolve Pass Finished ---")