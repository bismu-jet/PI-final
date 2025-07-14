import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

# --- Importing custom modules for the solver ---
from solver.problem import MIPProblem
from solver.utilities import setup_logger

# --- Setup the logger ---
# Creates a logger object to print progress and debug information for the cut generation.
logger = setup_logger()


def _generate_knapsack_cover_cuts(problem: MIPProblem, lp_solution: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Generates knapsack cover cuts for suitable constraints. 🎒

    A knapsack constraint has the form `Σ(a_i * x_i) <= b`, where all `x_i` are binary
    and all `a_i` are positive. A "cover" is a subset of variables `C` whose
    coefficients sum to more than `b`. If all variables in the cover were 1, the
    constraint would be violated. Therefore, at most `|C| - 1` of them can be 1,
    leading to the valid cut: `Σ(x_i for i in C) <= |C| - 1`.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        lp_solution (Dict[str, float]): The solution from the current node's LP relaxation.

    Returns:
        List[Dict[str, Any]]: A list of new, violated knapsack cover cuts.
    """
    new_cuts: List[Dict[str, Any]] = []
    MIN_VIOLATION: float = 1e-6
    # Create a set of binary variable names for efficient lookup.
    binary_vars: Set[str] = {v.VarName for v in problem.model.getVars() if v.VType == GRB.BINARY}

    # Iterate through constraints to find knapsack candidates.
    for constr in problem.model.getConstrs():
        # The cut applies to "less than or equal to" constraints.
        if constr.Sense != GRB.LESS_EQUAL:
            continue

        row: gp.LinExpr = problem.model.getRow(constr)
        rhs: float = constr.RHS
        
        # Check if the constraint fits the knapsack structure.
        is_knapsack_candidate: bool = True
        knapsack_vars: List[Tuple[str, float]] = []
        for i in range(row.size()):
            var: gp.Var = row.getVar(i)
            coeff: float = row.getCoeff(i)
            # All variables must be binary and all coefficients must be positive.
            if var.VarName not in binary_vars or coeff <= 0:
                is_knapsack_candidate = False
                break
            knapsack_vars.append((var.VarName, coeff))

        if not is_knapsack_candidate:
            continue
        
        # --- Find a violated cover using a greedy heuristic ---
        potential_cover: List[str] = []
        current_weight: float = 0
        # Sort variables by their coefficient (weight) in descending order.
        sorted_by_coeff: List[Tuple[str, float]] = sorted(knapsack_vars, key=lambda x: x[1], reverse=True)

        # Greedily add variables to the cover until the RHS is exceeded.
        for var_name, coeff in sorted_by_coeff:
            potential_cover.append(var_name)
            current_weight += coeff
            if current_weight > rhs:
                # We found a cover C. Now check if the LP solution violates the corresponding cut.
                # The cut is: Σ(x_i for i in C) <= |C| - 1
                violation: float = sum(lp_solution.get(v, 0.0) for v in potential_cover) - (len(potential_cover) - 1)
                
                if violation > MIN_VIOLATION:
                    # If violated, create the cut dictionary.
                    cut_coeffs: Dict[str, float] = {v: 1.0 for v in potential_cover}
                    cut_rhs: float = float(len(potential_cover) - 1)
                    new_cut: Dict[str, Any] = {
                        'coeffs': cut_coeffs,
                        'sense': GRB.LESS_EQUAL,
                        'rhs': cut_rhs
                    }
                    new_cuts.append(new_cut)
                    logger.info(f"Generated Knapsack Cover Cut on '{constr.ConstrName}' with {len(potential_cover)} vars. Violation: {violation:.4f}")
                    # Stop after finding one violated cover per constraint to keep the process fast.
                    break
                    
    return new_cuts


def _generate_clique_cuts(problem: MIPProblem, lp_solution: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Generates clique cuts from a conflict graph. 🤝

    A conflict graph connects pairs of binary variables that cannot both be 1. Such a
    conflict is often represented by a constraint like `x_i + x_j <= 1`. A "clique"
    in this graph is a set of variables where every variable is in conflict with every
    other variable. This implies that at most one variable in the clique can be 1,
    leading to the valid cut: `Σ(x_i for i in Clique) <= 1`.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        lp_solution (Dict[str, float]): The solution from the current node's LP relaxation.

    Returns:
        List[Dict[str, Any]]: A list of new, violated clique cuts.
    """
    new_cuts: List[Dict[str, Any]] = []
    MIN_VIOLATION: float = 1e-6
    binary_vars: Set[str] = {v.VarName for v in problem.model.getVars() if v.VType == GRB.BINARY}
    
    # 1. Build the conflict graph from constraints of the form: x_i + x_j <= 1.
    conflict_graph: Dict[str, Set[str]] = defaultdict(set)
    for constr in problem.model.getConstrs():
        # Look for the specific structure: two variables, coefficients of 1, RHS of 1.
        if constr.Sense == GRB.LESS_EQUAL and constr.RHS == 1.0:
            row: gp.LinExpr = problem.model.getRow(constr)
            if row.size() == 2:
                var1, var2 = row.getVar(0), row.getVar(1)
                coeff1, coeff2 = row.getCoeff(0), row.getCoeff(1)

                if var1.VarName in binary_vars and var2.VarName in binary_vars and \
                   abs(coeff1 - 1.0) < 1e-9 and abs(coeff2 - 1.0) < 1e-9:
                    # Add an edge between the two conflicting variables.
                    conflict_graph[var1.VarName].add(var2.VarName)
                    conflict_graph[var2.VarName].add(var1.VarName)

    if not conflict_graph:
        return []

    # 2. Find maximal cliques using a greedy heuristic.
    # Start with nodes that have the most conflicts (highest degree).
    nodes_sorted_by_degree: List[str] = sorted(conflict_graph.keys(), key=lambda v: len(conflict_graph[v]), reverse=True)
    
    processed_nodes: Set[str] = set()
    for node in nodes_sorted_by_degree:
        if node in processed_nodes:
            continue

        # Greedily build a clique starting from the current node.
        clique: Set[str] = {node}
        # Candidates are neighbors of the starting node.
        candidates: Set[str] = conflict_graph[node] - processed_nodes
        
        for candidate in candidates:
            # A candidate can join the clique if it's connected to all existing members.
            if all(candidate in conflict_graph[member] for member in clique):
                clique.add(candidate)
        
        # 3. Check for violation and generate the cut: Σ(x_i for i in C) <= 1.
        if len(clique) > 1:
            violation: float = sum(lp_solution.get(v, 0.0) for v in clique) - 1.0
            if violation > MIN_VIOLATION:
                cut_coeffs: Dict[str, float] = {v: 1.0 for v in clique}
                new_cut: Dict[str, Any] = {
                    'coeffs': cut_coeffs,
                    'sense': GRB.LESS_EQUAL,
                    'rhs': 1.0
                }
                new_cuts.append(new_cut)
                logger.info(f"Generated Clique Cut with {len(clique)} vars. Violation: {violation:.4f}")

        # Mark nodes in the found clique as processed to avoid generating redundant sub-cliques.
        processed_nodes.update(clique)
            
    return new_cuts


def generate_gmi_cuts(solved_model: gp.Model, 
                      lp_result: Dict[str, Any], 
                      problem: MIPProblem) -> List[Dict[str, Any]]:
    """
    Generates Gomory Mixed-Integer (GMI) cuts from the simplex tableau. 📐

    GMI cuts are general-purpose cuts derived directly from the final simplex tableau
    of an LP relaxation. The process involves:
    1. Finding a basic variable that should be integer but is fractional.
    2. Using its corresponding row in the simplex tableau.
    3. Applying the GMI formula to this row to create a new valid inequality that
       cuts off the current fractional LP solution.

    Args:
        solved_model (gp.Model): The solved Gurobi model object from the LP relaxation.
        lp_result (Dict[str, Any]): The full result dictionary from the LP solve.
        problem (MIPProblem): The main MIP problem instance.

    Returns:
        List[Dict[str, Any]]: A list containing the single most violated GMI cut found.
    """
    solution: Optional[Dict[str, float]] = lp_result.get('solution')
    vbasis: Optional[Dict[str, int]] = lp_result.get('vbasis')
    cbasis: Optional[Dict[str, int]] = lp_result.get('cbasis')

    # This cut requires detailed information from the LP solve.
    if not all([solution, vbasis, cbasis, solved_model]):
        return []

    # --- Constants and Initialization ---
    INT_TOL, ZERO_TOL, MIN_VIOLATION = 1e-6, 1e-9, 1e-6
    best_cut: Optional[Dict[str, Any]] = None
    max_violation: float = 0.0
    
    all_vars: List[gp.Var] = solved_model.getVars()
    var_names: List[str] = [v.VarName for v in all_vars]
    constraints: List[gp.Constr] = solved_model.getConstrs()
    num_vars, num_constrs = len(var_names), len(constraints)
    var_status: Dict[str, int] = {name: vbasis.get(name) for name in var_names}
    
    # --- Build Simplex Tableau Components ---
    # Construct the full constraint matrix A, including slack variables.
    A_sparse: csr_matrix = solved_model.getA()
    A_full_sparse: csr_matrix = csr_matrix(np.hstack([A_sparse.toarray(), np.identity(num_constrs)]))
    b_vector: np.ndarray = np.array([c.RHS for c in constraints])

    # Identify the indices of all basic variables (both structural and slack).
    basic_indices: List[int] = [i for i, name in enumerate(var_names) if var_status.get(name) == GRB.BASIC]
    for i, constr in enumerate(constraints):
        if cbasis.get(constr.ConstrName) == GRB.BASIC:
            basic_indices.append(num_vars + i)
    
    if len(basic_indices) != num_constrs:
        logger.warning(f"Basis size inconsistency. Expected: {num_constrs}, Got: {len(basic_indices)}. Cannot generate cuts.")
        return []

    # The basis matrix B consists of the columns of A corresponding to basic variables.
    B: np.ndarray = A_full_sparse[:, basic_indices].toarray()

    # Iterate through fractional integer basic variables to find a source row for a cut.
    for var_name in problem.integer_variable_names:
        if var_status.get(var_name) == GRB.BASIC and abs(solution[var_name] - round(solution[var_name])) > INT_TOL:
            source_var_index: int = var_names.index(var_name)
            try: 
                source_in_basis_pos: int = basic_indices.index(source_var_index)
            except ValueError: 
                continue

            # --- Calculate the Tableau Row ---
            # Solve B.T * alpha = e_k to find the simplex multipliers for this row.
            e: np.ndarray = np.zeros(num_constrs); e[source_in_basis_pos] = 1.0
            try: 
                alpha: np.ndarray = np.linalg.solve(B.T, e)
            except np.linalg.LinAlgError: 
                continue

            # The tableau row is alpha * A_full and its RHS is alpha * b.
            tableau_coeffs: np.ndarray = alpha @ A_full_sparse.toarray()
            tableau_rhs: float = alpha @ b_vector

            # Sanity check: the calculated RHS should match the variable's fractional value.
            if abs(tableau_rhs - solution[var_name]) > INT_TOL: 
                continue

            f0: float = tableau_rhs - math.floor(tableau_rhs)
            if f0 < INT_TOL or f0 > (1.0 - INT_TOL): 
                continue

            # --- Construct the GMI Cut using the formula ---
            cut_lhs_coeffs: Dict[str, float] = {}
            cut_rhs_adjustment: float = 0.0
            # Iterate through the non-basic variables to build the cut expression.
            for j in range(num_vars):
                var_j_name: str = var_names[j]
                # Skip basic variables. Gurobi status for non-basic at LB is -1, at UB is -2.
                if var_status.get(var_j_name) not in [-1, -2]: 
                    continue
                
                a_bar_j: float = tableau_coeffs[j]
                f_j: float = a_bar_j - math.floor(a_bar_j)
                coeff: float = 0.0

                # Apply the GMI formula based on whether the non-basic var is at its lower or upper bound.
                if var_status.get(var_j_name) == -1: # At Lower Bound
                    if f_j > f0 + ZERO_TOL: coeff = (f_j - f0) / (1.0 - f0)
                elif var_status.get(var_j_name) == -2: # At Upper Bound
                    if f_j < f0 - ZERO_TOL: coeff = f_j / f0
                
                if abs(coeff) > ZERO_TOL:
                    if var_status.get(var_j_name) == -1: # Non-basic at LB
                        cut_lhs_coeffs[var_j_name] = coeff
                    else: # Non-basic at UB
                        cut_lhs_coeffs[var_j_name] = -coeff
                        cut_rhs_adjustment += coeff * all_vars[j].UB
            
            if not cut_lhs_coeffs: 
                continue
            
            # Check for violation and save the most violated cut found so far.
            final_cut_rhs: float = 1.0 - cut_rhs_adjustment
            cut_activity: float = sum(c * solution.get(name, 0) for name, c in cut_lhs_coeffs.items())
            violation: float = cut_activity - final_cut_rhs

            if violation > MIN_VIOLATION and violation > max_violation:
                max_violation, best_cut = violation, {'coeffs': cut_lhs_coeffs, 'sense': GRB.GREATER_EQUAL, 'rhs': final_cut_rhs}
    
    if best_cut:
        logger.info(f"Generated 1 GMI cut with {len(best_cut['coeffs'])} terms and violation of {max_violation:.6f}.")
        return [best_cut]
    return []


def generate_all_cuts(problem: MIPProblem,
                      lp_result: Dict[str, Any],
                      active_cuts: List[Dict[str, Any]] = [],
                      local_constraints: List[Tuple[str, str, float]] = []) -> List[Dict[str, Any]]:
    """
    Orchestrates the generation of all implemented cut types. 🔪

    This master function calls each individual cut generator and collects all new,
    violated cuts into a single list.

    Args:
        problem (MIPProblem): The main MIP problem instance.
        lp_result (Dict[str, Any]): The full result dictionary from the LP solve.
        active_cuts (List[Dict[str, Any]]): List of cuts already applied to this node. (unused in this version)
        local_constraints (List[Tuple[str, str, float]]): Local B&B constraints. (unused in this version)

    Returns:
        List[Dict[str, Any]]: A list of all newly generated cuts.
    """
    logger.debug("--- Starting Cut Generation ---")
    all_new_cuts: List[Dict[str, Any]] = []
    lp_solution: Optional[Dict[str, float]] = lp_result.get('solution')

    if not lp_solution:
        logger.warning("Cannot generate cuts without an LP solution.")
        return []
    
    # --- Get the Gurobi model object from the result dictionary ---
    # This is needed for the GMI cut generator.
    solved_model: Optional[gp.Model] = lp_result.get('model')

    # 1. Generate structural cuts (which don't require the simplex tableau).
    knapsack_cuts: List[Dict[str, Any]] = _generate_knapsack_cover_cuts(problem, lp_solution)
    if knapsack_cuts:
        all_new_cuts.extend(knapsack_cuts)

    clique_cuts: List[Dict[str, Any]] = _generate_clique_cuts(problem, lp_solution)
    if clique_cuts:
        all_new_cuts.extend(clique_cuts)

    # 2. Generate general-purpose cuts from the tableau.
    if solved_model:
        gmi_cuts: List[Dict[str, Any]] = generate_gmi_cuts(solved_model, lp_result, problem)
        if gmi_cuts:
            all_new_cuts.extend(gmi_cuts)
    else:
        logger.debug("Skipping GMI cuts because the solved model was not provided.")

    if all_new_cuts:
        logger.debug(f"--- Total of {len(all_new_cuts)} new cuts generated for this node. ---")
            
    return all_new_cuts