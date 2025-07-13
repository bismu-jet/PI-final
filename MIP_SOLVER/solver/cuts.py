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
    
    # --- We will now use the more stable Gomory cut generator ---
    gomory_cuts = generate_stable_gomory_cuts(problem, lp_result)
    all_cuts.extend(gomory_cuts)
    
    return all_cuts

def generate_stable_gomory_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    --- REVISED: Generates numerically stable Gomory Mixed-Integer (GMI) cuts. ---
    
    This version includes checks for cut violation and dynamism to avoid
    adding weak or numerically unstable cuts to the model.
    """
    solution = lp_result.get('solution')
    if not solution or not lp_result.get('vbasis') or not lp_result.get('cbasis'):
        return []

    # --- Parameters for cut quality control ---
    MIN_VIOLATION = 1e-4
    MAX_DYNAMISM = 1e4

    source_row_info = None
    # Find a fractional integer variable that is basic in the LP solution
    for var_name in problem.integer_variable_names:
        if lp_result['vbasis'].get(var_name) == GRB.BASIC:
            val = solution[var_name]
            if abs(val - round(val)) > 1e-6:
                source_row_info = {'name': var_name, 'value': val}
                logger.debug(f"Found fractional basic integer variable '{var_name}' (value: {val:.4f}) to generate a cut.")
                break 
    
    if not source_row_info:
        return []

    temp_model = None
    try:
        # We build a temporary model to get the tableau row
        temp_model = problem.model.relax()
        
        local_constraints = lp_result.get('local_constraints', [])
        for var_name, sense, value in local_constraints:
            var = temp_model.getVarByName(var_name)
            if sense == '<=': temp_model.addConstr(var <= value)
            else: temp_model.addConstr(var >= value)
        temp_model.update()

        # Extract basis information to compute the tableau
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
        
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            logger.warning("Basis matrix is singular. Skipping Gomory cut generation.")
            return []

        source_var_index = var_names.index(source_row_info['name'])
        source_in_basis_pos = basic_indices.index(source_var_index)
        alpha = B_inv[source_in_basis_pos]
        tableau_rhs = alpha @ b
        tableau_coeffs = alpha @ A_full_sparse.toarray()

        f0 = tableau_rhs - math.floor(tableau_rhs)
        if f0 < 1e-6 or f0 > (1.0 - 1e-6): 
            return []

        cut_lhs_coeffs = {}
        # Iterate through non-basic variables to build the cut expression
        for j, v_name in enumerate(var_names):
            if lp_result['vbasis'].get(v_name) == GRB.BASIC:
                continue

            var_j = temp_model.getVars()[j]
            a_bar_j = tableau_coeffs[j]
            fj = a_bar_j - math.floor(a_bar_j)
            
            is_at_upper = lp_result['vbasis'].get(v_name) == -2
            
            coeff = 0.0
            if fj <= f0:
                coeff = fj / f0
            else: 
                coeff = (1 - fj) / (1 - f0)
            
            if abs(coeff) > 1e-6:
                if is_at_upper:
                    cut_lhs_coeffs[v_name] = -coeff
                else:
                    cut_lhs_coeffs[v_name] = coeff
        
        if not cut_lhs_coeffs:
            return []

        # --- NEW: Quality checks for the generated cut ---
        
        # 1. Check for Violation
        cut_activity = sum(coeff * solution.get(var, 0) for var, coeff in cut_lhs_coeffs.items())
        violation = cut_activity - 1.0
        
        if violation < MIN_VIOLATION:
            logger.debug(f"Gomory cut rejected: Insufficient violation ({violation:.4f})")
            return []

        # 2. Check for Dynamism
        abs_coeffs = [abs(c) for c in cut_lhs_coeffs.values() if abs(c) > 1e-9]
        if not abs_coeffs or min(abs_coeffs) == 0:
            return []
        dynamism = max(abs_coeffs) / min(abs_coeffs)

        if dynamism > MAX_DYNAMISM:
            logger.debug(f"Gomory cut rejected: High dynamism ({dynamism:.2e})")
            return []
        
        logger.info(f"Generated a stable Gomory Cut with {len(cut_lhs_coeffs)} terms (violation: {violation:.4f}, dynamism: {dynamism:.2e}).")
        return [{'coeffs': cut_lhs_coeffs, 'sense': GRB.GREATER_EQUAL, 'rhs': 1.0}]

    except Exception as e:
        logger.error(f"An unexpected error occurred during Gomory cut generation: {e}", exc_info=True)
        return []
    finally:
        if temp_model:
            temp_model.dispose()
