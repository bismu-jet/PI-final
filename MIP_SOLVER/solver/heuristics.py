# solver/heuristics.py

import math
from typing import Dict, List, Optional, Tuple
from solver.problem import MIPProblem
from solver.gurobi_interface import solve_lp_relaxation

def find_initial_solution(
    problem: MIPProblem,
    starting_solution: Dict[str, float],
    initial_constraints: List[Tuple[str, str, float]],
    time_limit: Optional[float], # <-- ADD THIS ARGUMENT
    max_depth: int = 10
) -> Optional[Dict[str, float]]:
    """
    A simple diving heuristic to find an initial integer-feasible solution.
    This version now correctly passes the time limit.
    """

    def dive(current_solution, constraints, depth):
        # Base case: max depth reached or no time left
        if depth >= max_depth or (time_limit is not None and time_limit <= 0):
            return None

        # Check for integer feasibility
        is_feasible = all(
            abs(current_solution.get(var, 0) - round(current_solution.get(var, 0))) < 1e-6
            for var in problem.integer_variable_names
        )
        if is_feasible:
            return current_solution

        # Find the most fractional variable to dive on
        fractional_vars = [
            var for var in problem.integer_variable_names
            if abs(current_solution.get(var, 0) - round(current_solution.get(var, 0))) > 1e-6
        ]
        if not fractional_vars:
            return None # Should be feasible, but as a safeguard

        branch_var = max(fractional_vars, key=lambda v: abs(current_solution[v] - round(current_solution[v])))
        val = current_solution[branch_var]

        # Try diving down first
        constraints_down = constraints + [(branch_var, '<=', math.floor(val))]
        lp_result_down = solve_lp_relaxation(problem, constraints_down, time_limit=time_limit) # <-- PASS TIME LIMIT
        if lp_result_down.get('status') == 'OPTIMAL':
            solution = dive(lp_result_down['solution'], constraints_down, depth + 1)
            if solution:
                return solution

        # If that fails, try diving up
        constraints_up = constraints + [(branch_var, '>=', math.ceil(val))]
        lp_result_up = solve_lp_relaxation(problem, constraints_up, time_limit=time_limit) # <-- PASS TIME LIMIT
        if lp_result_up.get('status') == 'OPTIMAL':
            solution = dive(lp_result_up['solution'], constraints_up, depth + 1)
            if solution:
                return solution

        return None

    return dive(starting_solution, initial_constraints, 0)