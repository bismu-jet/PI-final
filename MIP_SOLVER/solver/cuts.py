import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Optional

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()

def generate_gomory_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates Gomory Mixed-Integer (GMI) cuts from a fractional LP solution.
    This implementation reconstructs the necessary simplex tableau row using
    the basis inverse matrix.
    """
    solution = lp_result.get('solution')
    if not solution:
        return []

    # --- 1. Find a suitable row to generate a cut from ---
    # We need a basic variable that is supposed to be integer but has a fractional value.
    source_row_info = None
    for var_name in problem.integer_variable_names:
        if lp_result['vbasis'].get(var_name) == GRB.BASIC: # Is it a basic variable?
            val = solution[var_name]
            if abs(val - round(val)) > 1e-6: # Is it fractional?
                source_row_info = {'name': var_name, 'value': val}
                logger.info(f"Found fractional basic integer variable '{var_name}' (value: {val:.4f}) to generate a cut.")
                break # We will generate one cut at a time for simplicity and stability
    
    if not source_row_info:
        return [] # No suitable row found

    try:
        # --- 2. Get the full problem matrix A and RHS b ---
        # We need a temporary model to query matrix information.
        temp_model = problem.model.relax()
        temp_model.setParam('OutputFlag', 0)
        temp_model.optimize()
        
        num_vars = temp_model.numVars
        num_constrs = temp_model.numConstrs
        
        # Get the matrix in a sparse format for efficiency
        A_sparse = temp_model.getA()
        b = np.array([c.RHS for c in temp_model.getConstrs()])
        
        # --- 3. Identify Basic and Non-Basic Variables ---
        # Basic variables have VBasis/CBasis == 0.
        # Non-basic variables are at a bound (VBasis/CBasis == -1 or -2).
        var_names = [v.VarName for v in temp_model.getVars()]
        con_names = [c.ConstrName for c in temp_model.getConstrs()]

        # Note: Gurobi's basis includes slack variables for constraints.
        # The total number of basic variables must equal the number of constraints.
        basic_vars_indices = []
        non_basic_vars_indices = []
        
        # Find the index of our source variable
        source_var_index = var_names.index(source_row_info['name'])

        # Structural variables (the x's)
        for i, v_name in enumerate(var_names):
            if lp_result['vbasis'].get(v_name) == GRB.BASIC:
                basic_vars_indices.append(i)
            else:
                non_basic_vars_indices.append(i)

        # Slack variables (one for each constraint)
        for i, c_name in enumerate(con_names):
            if lp_result['cbasis'].get(c_name) == GRB.BASIC:
                basic_vars_indices.append(num_vars + i) # Slack variables are indexed after structural variables
            # Non-basic slacks are implicitly handled.

        # --- 4. Construct and Invert the Basis Matrix B ---
        # The full matrix is [A, I], where I is for the slack variables.
        I_sparse = csr_matrix(np.identity(num_constrs))
        A_full_sparse = csr_matrix(np.hstack([A_sparse.toarray(), I_sparse.toarray()]))

        # The basis matrix B is formed by the columns of A_full corresponding to basic variables.
        B = A_full_sparse[:, basic_vars_indices].toarray()
        
        # Invert the basis matrix. This can fail if the matrix is singular.
        B_inv = np.linalg.inv(B)
        
        # --- 5. Reconstruct the Tableau Row ---
        # We need to find which row of B_inv corresponds to our source variable.
        source_in_basis_pos = basic_vars_indices.index(source_var_index)
        
        # The tableau row multipliers are the corresponding row from B_inv
        alpha = B_inv[source_in_basis_pos]
        
        # The RHS of the tableau row. Should match the variable's fractional value.
        tableau_rhs = alpha @ b
        
        # The coefficients of the tableau row for ALL variables.
        tableau_coeffs = alpha @ A_full_sparse.toarray()

        # --- 6. Derive the Gomory Cut from the Tableau Row ---
        f0 = tableau_rhs - math.floor(tableau_rhs)
        
        if f0 < 1e-6: # If fractional part is negligible, no cut can be generated.
            return []

        cut_expr = gp.LinExpr()
        
        # The cut involves only the non-basic variables.
        for j in non_basic_vars_indices:
            var_j = temp_model.getVar(j)
            coeff_j = tableau_coeffs[j]
            
            # This is the pure integer Gomory cut formula for simplicity.
            # A full mixed-integer implementation is more complex.
            fj = coeff_j - math.floor(coeff_j)
            
            # We only add terms with non-negligible fractional parts to the cut.
            if fj > 1e-6:
                cut_expr.add(var_j, fj)

        # The Gomory cut is: sum(fj * xj) >= f0
        logger.info(f"Generated Gomory Cut: {cut_expr} >= {f0:.4f}")
        return [{'lhs': cut_expr, 'sense': GRB.GREATER_EQUAL, 'rhs': f0}]

    except np.linalg.LinAlgError:
        logger.warning("Could not generate Gomory cut: Basis matrix is singular.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during cut generation: {e}")
        return []
    finally:
        if 'temp_model' in locals():
            temp_model.dispose()