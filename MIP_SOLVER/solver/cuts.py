import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Optional

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()

def generate_all_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Master function to generate all types of cuts.
    It calls individual cut generators and aggregates the results.
    """
    all_cuts = []
    
    gomory_cuts = generate_gomory_cuts(problem, lp_result)
    all_cuts.extend(gomory_cuts)
    
    mir_cuts = generate_mir_cuts(problem, lp_result)
    all_cuts.extend(mir_cuts)
    
    return all_cuts

def generate_mir_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates Mixed-Integer Rounding (MIR) cuts from single problem constraints.
    This is a rewritten, more robust version with extensive logging.
    """
    logger.info("Attempting to generate MIR cuts...")
    solution = lp_result.get('solution')
    if not solution:
        return []

    generated_cuts = []
    
    for constr in problem.model.getConstrs():
        if constr.Sense != GRB.LESS_EQUAL:
            continue

        row = problem.model.getRow(constr)
        b_original = constr.RHS
        
        integer_vars_coeffs = {}
        continuous_vars_coeffs = {}

        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            if var.VarName in problem.integer_variable_names:
                integer_vars_coeffs[var.VarName] = coeff
            else:
                continuous_vars_coeffs[var.VarName] = coeff

        if not integer_vars_coeffs:
            continue
        
        logger.debug(f"--- Analyzing constraint '{constr.ConstrName}' for MIR cut ---")
        
        b_prime = b_original
        for var_name, coeff in continuous_vars_coeffs.items():
            var = problem.model.getVarByName(var_name)
            if coeff > 0:
                if var.LB == -GRB.INFINITY:
                    b_prime = None; break
                b_prime -= coeff * var.LB
            else:
                if var.UB == GRB.INFINITY:
                    b_prime = None; break
                b_prime -= coeff * var.UB
        
        if b_prime is None:
            logger.debug(f"Skipping '{constr.ConstrName}': Unbounded continuous variable part.")
            continue

        b_floor = math.floor(b_prime)
        f0 = b_prime - b_floor
        
        logger.debug(f"Original RHS b={b_original:.4f}, Adjusted RHS b'={b_prime:.4f}, f0={f0:.4f}")

        if f0 < 1e-6 or f0 > (1.0 - 1e-6):
            logger.debug("Skipping: f0 is negligible.")
            continue

        cut_lhs_coeffs = {}
        for var_name, coeff in integer_vars_coeffs.items():
            fi = coeff - math.floor(coeff)
            
            # --- CORRECTED MIR FORMULA ---
            # This is the standard, correct formula for the MIR coefficient.
            new_coeff = math.floor(coeff) + max(0, fi - f0) / (1 - f0)
            # --- END CORRECTION ---
            
            cut_lhs_coeffs[var_name] = new_coeff
            logger.debug(f"  Var '{var_name}': Original coeff={coeff:.4f}, fi={fi:.4f}, New MIR coeff={new_coeff:.4f}")

        current_lhs_val = 0
        for var_name, coeff in cut_lhs_coeffs.items():
            current_lhs_val += coeff * solution.get(var_name, 0)
        
        logger.debug(f"Violation Check: Cut LHS value = {current_lhs_val:.4f}, Cut RHS = {b_floor:.4f}")
        
        if current_lhs_val > b_floor + 1e-6:
            logger.info(f"SUCCESS: Generated a violated MIR cut from constraint '{constr.ConstrName}'.")
            generated_cuts.append({
                'coeffs': cut_lhs_coeffs,
                'sense': GRB.LESS_EQUAL,
                'rhs': b_floor
            })
            return generated_cuts

    logger.info(f"Finished MIR cut generation attempt. Found {len(generated_cuts)} cuts.")
    return generated_cuts


def generate_gomory_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates Gomory Mixed-Integer (GMI) cuts from a fractional LP solution.
    This version correctly handles non-basic variables at both lower and upper bounds.
    """
    solution = lp_result.get('solution')
    if not solution or not lp_result.get('vbasis') or not lp_result.get('cbasis'):
        return []

    source_row_info = None
    for var_name in problem.integer_variable_names:
        if lp_result['vbasis'].get(var_name) == GRB.BASIC:
            val = solution[var_name]
            if abs(val - round(val)) > 1e-6:
                source_row_info = {'name': var_name, 'value': val}
                logger.info(f"Found fractional basic integer variable '{var_name}' (value: {val:.4f}) to generate a cut.")
                break 
    
    if not source_row_info:
        return []

    temp_model = None
    try:
        temp_model = problem.model.relax()
        
        local_constraints = lp_result.get('local_constraints', [])
        for var_name, sense, value in local_constraints:
            var = temp_model.getVarByName(var_name)
            if sense == '<=': temp_model.addConstr(var <= value)
            else: temp_model.addConstr(var >= value)
        temp_model.update()

        A_sparse = temp_model.getA()
        b = np.array([c.RHS for c in temp_model.getConstrs()])
        var_names = [v.VarName for v in temp_model.getVars()]
        con_names = [c.ConstrName for c in temp_model.getConstrs()]
        num_vars = len(var_names)
        num_constrs = len(con_names)

        basic_indices = [i for i, v_name in enumerate(var_names) if lp_result['vbasis'].get(v_name) == GRB.BASIC]
        basic_indices += [num_vars + i for i, c_name in enumerate(con_names) if lp_result['cbasis'].get(c_name) == GRB.BASIC]
        
        if len(basic_indices) != num_constrs:
             logger.warning("Basis size mismatch. Skipping Gomory cut generation.")
             return []

        A_full_sparse = csr_matrix(np.hstack([A_sparse.toarray(), np.identity(num_constrs)]))
        B = A_full_sparse[:, basic_indices].toarray()
        B_inv = np.linalg.inv(B)

        source_var_index = var_names.index(source_row_info['name'])
        source_in_basis_pos = basic_indices.index(source_var_index)
        alpha = B_inv[source_in_basis_pos]
        tableau_rhs = alpha @ b
        tableau_coeffs = alpha @ A_full_sparse.toarray()

        f0 = tableau_rhs - math.floor(tableau_rhs)
        if f0 < 1e-6 or f0 > (1.0 - 1e-6): 
            return []

        cut_lhs = gp.LinExpr()
        cut_rhs = 1.0

        for j, v_name in enumerate(var_names):
            if lp_result['vbasis'].get(v_name) == GRB.BASIC:
                continue

            var_j = temp_model.getVars()[j]
            
            a_bar_j = tableau_coeffs[j]
            fj = a_bar_j - math.floor(a_bar_j)
            
            is_at_upper = lp_result['vbasis'].get(v_name) == -2
            
            if fj <= f0:
                if is_at_upper:
                    cut_lhs.add(var_j, -fj / f0)
                    cut_rhs -= (fj / f0) * var_j.UB
                else:
                    cut_lhs.add(var_j, fj / f0)
            else: 
                coeff = (1 - fj) / (1 - f0)
                if is_at_upper:
                    cut_lhs.add(var_j, -coeff)
                    cut_rhs -= coeff * var_j.UB
                else:
                    cut_lhs.add(var_j, coeff)
        
        if cut_lhs.size() == 0:
            return []

        logger.info(f"Generated refined Gomory Cut with {cut_lhs.size()} terms.")
        return [{'coeffs': {cut_lhs.getVar(i).VarName: cut_lhs.getCoeff(i) for i in range(cut_lhs.size())}, 
                 'sense': GRB.GREATER_EQUAL, 
                 'rhs': cut_rhs}]

    except Exception as e:
        logger.error(f"An unexpected error occurred during Gomory cut generation: {e}")
        return []
    finally:
        if temp_model:
            temp_model.dispose()
