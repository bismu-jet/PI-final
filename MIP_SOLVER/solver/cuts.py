import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()


def _generate_knapsack_cover_cuts(problem: MIPProblem, lp_solution: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    --- NEW: Generates Knapsack Cover Cuts. ---

    Identifies constraints of the form Sum(a_i * x_i) <= b, where x_i are binary
    variables and a_i > 0. It then looks for a "cover" - a set of variables C
    such that Sum(a_i for i in C) > b. For any such cover, the valid cut
    Sum(x_i for i in C) <= |C| - 1 can be generated.

    This implementation uses a simple greedy approach to find violated covers.
    """
    new_cuts = []
    MIN_VIOLATION = 1e-6
    binary_vars = {v.VarName for v in problem.model.getVars() if v.VType == GRB.BINARY}

    for constr in problem.model.getConstrs():
        # We are looking for standard knapsack constraints: Σaᵢxᵢ ≤ b
        if constr.Sense != GRB.LESS_EQUAL:
            continue

        row = problem.model.getRow(constr)
        rhs = constr.RHS
        
        # Check if this is a candidate for a knapsack cut
        is_knapsack_candidate = True
        knapsack_vars = []
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            # All variables must be binary and all coefficients positive
            if var.VarName not in binary_vars or coeff <= 0:
                is_knapsack_candidate = False
                break
            knapsack_vars.append((var.VarName, coeff))

        if not is_knapsack_candidate:
            continue
        
        # --- Simple Greedy Cover Finder ---
        potential_cover = []
        current_weight = 0
        # Sort variables by coefficient to build the cover
        sorted_by_coeff = sorted(knapsack_vars, key=lambda x: x[1], reverse=True)

        for var_name, coeff in sorted_by_coeff:
            potential_cover.append(var_name)
            current_weight += coeff
            if current_weight > rhs:
                # We found a cover C = potential_cover
                # The cut is Σ(x_i for i in C) <= |C| - 1
                
                # Check if this cut is violated by the current LP solution
                violation = sum(lp_solution.get(v, 0.0) for v in potential_cover) - (len(potential_cover) - 1)
                
                if violation > MIN_VIOLATION:
                    cut_coeffs = {v: 1.0 for v in potential_cover}
                    cut_rhs = float(len(potential_cover) - 1)
                    new_cut = {
                        'coeffs': cut_coeffs,
                        'sense': GRB.LESS_EQUAL,
                        'rhs': cut_rhs
                    }
                    new_cuts.append(new_cut)
                    logger.info(f"Generated Knapsack Cover Cut on '{constr.ConstrName}' with {len(potential_cover)} vars. Violation: {violation:.4f}")
                    # Stop after finding one cover for this constraint for simplicity
                    break
                    
    return new_cuts


def _generate_clique_cuts(problem: MIPProblem, lp_solution: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    --- NEW: Generates Clique Cuts. ---

    Finds sets of binary variables (cliques) where at most one variable in the
    set can be equal to 1. It does this by building a conflict graph from
    constraints of the form x_i + x_j <= 1. For each clique C found, it checks
    if the cut Sum(x_i for i in C) <= 1 is violated.
    """
    new_cuts = []
    MIN_VIOLATION = 1e-6
    binary_vars = {v.VarName for v in problem.model.getVars() if v.VType == GRB.BINARY}
    
    # 1. Build the conflict graph
    conflict_graph = defaultdict(set)
    for constr in problem.model.getConstrs():
        # A common structure for conflicts is x_i + x_j <= 1
        if constr.Sense == GRB.LESS_EQUAL and constr.RHS == 1.0:
            row = problem.model.getRow(constr)
            if row.size() == 2:
                var1 = row.getVar(0)
                var2 = row.getVar(1)
                coeff1 = row.getCoeff(0)
                coeff2 = row.getCoeff(1)

                # If we have 1*x_i + 1*x_j <= 1, they are in conflict
                if var1.VarName in binary_vars and var2.VarName in binary_vars and \
                   abs(coeff1 - 1.0) < 1e-9 and abs(coeff2 - 1.0) < 1e-9:
                    conflict_graph[var1.VarName].add(var2.VarName)
                    conflict_graph[var2.VarName].add(var1.VarName)

    if not conflict_graph:
        return []

    # 2. Find maximal cliques using a greedy heuristic
    # Start with nodes that have the most conflicts (highest degree)
    nodes_sorted_by_degree = sorted(conflict_graph.keys(), key=lambda v: len(conflict_graph[v]), reverse=True)
    
    processed_nodes = set()
    for node in nodes_sorted_by_degree:
        if node in processed_nodes:
            continue

        # Start a new clique with the current node
        clique = {node}
        # Candidates are neighbors of the current node that haven't been processed yet
        candidates = conflict_graph[node] - processed_nodes
        
        for candidate in candidates:
            # A candidate can be added if it's connected to all nodes already in the clique
            is_fully_connected = all(candidate in conflict_graph[member] for member in clique)
            if is_fully_connected:
                clique.add(candidate)
        
        # 3. Check for violation and generate cut
        if len(clique) > 1:
            violation = sum(lp_solution.get(v, 0.0) for v in clique) - 1.0
            if violation > MIN_VIOLATION:
                cut_coeffs = {v: 1.0 for v in clique}
                new_cut = {
                    'coeffs': cut_coeffs,
                    'sense': GRB.LESS_EQUAL,
                    'rhs': 1.0
                }
                new_cuts.append(new_cut)
                logger.info(f"Generated Clique Cut with {len(clique)} vars. Violation: {violation:.4f}")

        processed_nodes.update(clique)
        
    return new_cuts


def generate_gmi_cuts(problem: MIPProblem, lp_result: Dict[str, Any], active_cuts: List[Dict[str, Any]], local_constraints: List[Tuple[str, str, float]]) -> List[Dict[str, Any]]:
    """
    --- VERSÃO FINAL CORRIGIDA ---
    Gera os cortes Gomory Mixed-Integer (GMI) mais violados.
    """
    solution = lp_result.get('solution')
    vbasis = lp_result.get('vbasis')
    cbasis = lp_result.get('cbasis')

    if not all([solution, vbasis, cbasis]):
        return []

    INT_TOL, ZERO_TOL, MIN_VIOLATION = 1e-6, 1e-9, 1e-6
    best_cut, max_violation = None, 0.0
    temp_model = None

    try:
        temp_model = problem.model.copy().relax()
        
        for constr_data in local_constraints:
            var = temp_model.getVarByName(constr_data[0])
            if var is not None:
                if constr_data[1] == '<=': temp_model.addConstr(var <= constr_data[2])
                else: temp_model.addConstr(var >= constr_data[2])

        for i, cut in enumerate(active_cuts):
            expr = gp.LinExpr([(temp_model.getVarByName(v), c) for v, c in cut['coeffs'].items() if temp_model.getVarByName(v) is not None])
            temp_model.addConstr(expr, sense=cut['sense'], rhs=cut['rhs'], name=f"active_cut_{i}")
        
        temp_model.update()

        all_vars = temp_model.getVars()
        var_names = [v.VarName for v in all_vars]
        constraints = temp_model.getConstrs()
        num_vars, num_constrs = len(var_names), len(constraints)
        var_status = {name: vbasis.get(name) for name in var_names}
        
        A_sparse = temp_model.getA()
        A_full_sparse = csr_matrix(np.hstack([A_sparse.toarray(), np.identity(num_constrs)]))
        b_vector = np.array([c.RHS for c in constraints])

        # --- CORREÇÃO FINAL NA CONSTRUÇÃO DA BASE ---
        basic_indices = [i for i, name in enumerate(var_names) if var_status.get(name) == GRB.BASIC]
        
        constr_names_from_basis = list(cbasis.keys())
        constr_map = {name: i for i, name in enumerate(constr_names_from_basis)}

        for i, constr in enumerate(constraints):
            if cbasis.get(constr.ConstrName) == GRB.BASIC:
                 basic_indices.append(num_vars + i)
        
        if len(basic_indices) != num_constrs:
            logger.warning(f"Inconsistência no tamanho da base. Esperado: {num_constrs}, Obtido: {len(basic_indices)}. Não é possível gerar cortes.")
            return []

        B = A_full_sparse[:, basic_indices].toarray()

        for var_name in problem.integer_variable_names:
            if var_status.get(var_name) == GRB.BASIC and abs(solution[var_name] - round(solution[var_name])) > INT_TOL:
                source_var_index = var_names.index(var_name)
                try: source_in_basis_pos = basic_indices.index(source_var_index)
                except ValueError: continue

                e = np.zeros(num_constrs); e[source_in_basis_pos] = 1.0
                try: alpha = np.linalg.solve(B.T, e)
                except np.linalg.LinAlgError: continue

                tableau_coeffs = alpha @ A_full_sparse.toarray()
                tableau_rhs = alpha @ b_vector

                if abs(tableau_rhs - solution[var_name]) > INT_TOL: continue

                f0 = tableau_rhs - math.floor(tableau_rhs)
                if f0 < INT_TOL or f0 > (1.0 - INT_TOL): continue

                cut_lhs_coeffs, cut_rhs_adjustment = {}, 0.0
                for j in range(num_vars):
                    if var_status.get(var_names[j]) == GRB.BASIC: continue
                    var_j_name = var_names[j]
                    if var_status.get(var_j_name) not in [0, -2]: continue

                    a_bar_j = tableau_coeffs[j]; f_j = a_bar_j - math.floor(a_bar_j)
                    coeff = 0.0
                    
                    if var_status.get(var_j_name) == 0: # At Lower Bound
                        if f_j > f0 + ZERO_TOL: coeff = (f_j - f0) / (1.0 - f0)
                    elif var_status.get(var_j_name) == -2: # At Upper Bound
                        if f_j < f0 - ZERO_TOL: coeff = f_j / f0
                    
                    if abs(coeff) > ZERO_TOL:
                        if var_status.get(var_j_name) == 0:
                            cut_lhs_coeffs[var_j_name] = coeff
                        else:
                            cut_lhs_coeffs[var_j_name] = -coeff
                            cut_rhs_adjustment += coeff * all_vars[j].UB
                
                if not cut_lhs_coeffs: continue

                final_cut_rhs = 1.0 - cut_rhs_adjustment
                cut_activity = sum(c * solution.get(name, 0) for name, c in cut_lhs_coeffs.items())
                violation = cut_activity - final_cut_rhs

                if violation < -MIN_VIOLATION:
                    abs_violation = abs(violation)
                    if abs_violation > max_violation:
                        max_violation, best_cut = abs_violation, {'coeffs': cut_lhs_coeffs, 'sense': GRB.GREATER_EQUAL, 'rhs': final_cut_rhs}
        
        if best_cut:
            logger.info(f"Gerado 1 Corte GMI com {len(best_cut['coeffs'])} termos e violação de {max_violation:.6f}.")
            return [best_cut]
        return []
    finally:
        if temp_model: temp_model.dispose()


def generate_all_cuts(problem: MIPProblem, lp_result: Dict[str, Any], active_cuts: List[Dict[str, Any]] = [], local_constraints: List[Tuple[str, str, float]] = []) -> List[Dict[str, Any]]:
    """
    --- REVISED: Orchestrates the generation of all supported cut types. ---
    """
    logger.debug("--- Starting Cut Generation ---")
    all_new_cuts = []
    lp_solution = lp_result.get('solution')

    if not lp_solution:
        logger.warning("Cannot generate cuts without an LP solution.")
        return []

    # --- 1. Generate simpler, structural cuts first ---
    knapsack_cuts = _generate_knapsack_cover_cuts(problem, lp_solution)
    if knapsack_cuts:
        all_new_cuts.extend(knapsack_cuts)

    clique_cuts = _generate_clique_cuts(problem, lp_solution)
    if clique_cuts:
        all_new_cuts.extend(clique_cuts)

    # --- 2. Generate more general-purpose, complex cuts ---
    gmi_cuts = generate_gmi_cuts(problem, lp_result, active_cuts, local_constraints)
    if gmi_cuts:
        all_new_cuts.extend(gmi_cuts)

    logger.debug(f"--- Total of {len(all_new_cuts)} new cuts generated for this node. ---")
    return all_new_cuts