import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Optional

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()


def generate_clique_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    --- NEW: Generates Clique Cuts from mutual incompatibilities. ---
    
    Identifies constraints of the form x + y <= 1 to build an incompatibility graph.
    It then finds maximal cliques in this graph to generate cuts of the form
    sum_{j in Clique} x_j <= 1.
    """
    generated_cuts = []
    solution = lp_result.get('solution')
    if not solution:
        return []

    model = problem.model
    binary_vars = {v.VarName for v in model.getVars() if v.VType == GRB.BINARY}
    
    # 1. Build the incompatibility graph
    graph = {v_name: set() for v_name in binary_vars}
    for constr in model.getConstrs():
        if constr.Sense == GRB.LESS_EQUAL and constr.RHS == 1.0:
            row = model.getRow(constr)
            if row.size() == 2:
                var1, var2 = row.getVar(0), row.getVar(1)
                coeff1, coeff2 = row.getCoeff(0), row.getCoeff(1)
                
                # Check for the classic x + y <= 1 structure
                if var1.VarName in binary_vars and var2.VarName in binary_vars and \
                   abs(coeff1 - 1.0) < 1e-6 and abs(coeff2 - 1.0) < 1e-6:
                    graph[var1.VarName].add(var2.VarName)
                    graph[var2.VarName].add(var1.VarName)

    # 2. Find maximal cliques using a greedy heuristic
    found_cliques = set()
    # Sort variables by their LP solution value to prioritize forming cliques with fractional parts
    sorted_vars = sorted(list(binary_vars), key=lambda v: solution.get(v, 0), reverse=True)
    
    for var_name in sorted_vars:
        if var_name in {v for clique in found_cliques for v in clique}:
            continue # Skip if already part of a found clique

        clique = {var_name}
        # Find candidates that are incompatible with the current clique members
        candidates = graph[var_name]
        for member in clique:
            candidates = candidates.intersection(graph[member])

        while candidates:
            # Greedily add the candidate with the highest LP value
            best_candidate = max(candidates, key=lambda v: solution.get(v, 0))
            clique.add(best_candidate)
            candidates.remove(best_candidate)
            # Update candidates to be incompatible with the new member
            candidates = candidates.intersection(graph[best_candidate])
        
        if len(clique) > 1:
            # Use a frozenset as a key to store unique cliques
            found_cliques.add(frozenset(clique))
            
    # 3. Generate cuts for violated cliques
    for clique in found_cliques:
        if len(clique) < 2:
            continue
        
        # Check for violation: sum(x_j) > 1
        cut_activity = sum(solution.get(var, 0) for var in clique)
        if cut_activity > 1.0 + 1e-6:
            logger.info(f"Generated a Clique Cut with {len(clique)} terms.")
            cut_coeffs = {var: 1.0 for var in clique}
            generated_cuts.append({'coeffs': cut_coeffs, 'sense': GRB.LESS_EQUAL, 'rhs': 1.0})
            
    return generated_cuts


def generate_cover_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates Cover Cuts from knapsack-style constraints.
    """
    generated_cuts = []
    solution = lp_result.get('solution')
    if not solution:
        return []

    model = problem.model
    model.update()

    for constr in model.getConstrs():
        if constr.Sense != GRB.LESS_EQUAL:
            continue
        
        row = model.getRow(constr)
        b = constr.RHS
        
        is_candidate = True
        knapsack_vars = []
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            if var.VType != GRB.BINARY or coeff <= 0:
                is_candidate = False
                break
            knapsack_vars.append({'var': var.VarName, 'coeff': coeff})
            
        if not is_candidate or not knapsack_vars:
            continue
            
        knapsack_vars.sort(key=lambda item: solution.get(item['var'], 0), reverse=True)
        
        current_sum = 0
        cover = []
        for item in knapsack_vars:
            current_sum += item['coeff']
            cover.append(item['var'])
            if current_sum > b:
                break
        
        if current_sum <= b:
            continue
            
        cut_rhs = len(cover) - 1
        cut_activity = sum(solution.get(var_name, 0) for var_name in cover)
        
        if cut_activity > cut_rhs + 1e-6:
            logger.info(f"Generated a Cover Cut from constraint '{constr.ConstrName}' with {len(cover)} terms.")
            cut_coeffs = {var_name: 1.0 for var_name in cover}
            generated_cuts.append({'coeffs': cut_coeffs, 'sense': GRB.LESS_EQUAL, 'rhs': cut_rhs})

    return generated_cuts


def generate_all_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Master function to generate all types of cuts.
    It calls individual cut generators and aggregates the results.
    """
    all_cuts = []
    
    gomory_cuts = generate_stable_gomory_cuts(problem, lp_result)
    all_cuts.extend(gomory_cuts)
    
    cover_cuts = generate_cover_cuts(problem, lp_result)
    all_cuts.extend(cover_cuts)
    
    clique_cuts = generate_clique_cuts(problem, lp_result)
    all_cuts.extend(clique_cuts)
    
    return all_cuts


def generate_stable_gomory_cuts(problem: MIPProblem, lp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates numerically stable Gomory Mixed-Integer (GMI) cuts.
    """
    solution = lp_result.get('solution')
    if not solution or not lp_result.get('vbasis') or not lp_result.get('cbasis'):
        return []

    MIN_VIOLATION = 1e-4
    MAX_DYNAMISM = 1e4

    source_row_info = None
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
        
        cut_activity = sum(coeff * solution.get(var, 0) for var, coeff in cut_lhs_coeffs.items())
        violation = cut_activity - 1.0
        
        if violation < MIN_VIOLATION:
            logger.debug(f"Gomory cut rejected: Insufficient violation ({violation:.4f})")
            return []

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