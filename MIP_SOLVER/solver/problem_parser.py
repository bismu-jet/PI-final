# solver/problem_parser.py
import gurobipy as gp
from gurobipy import GRB
from solver.problem import MIPProblem, Variable, Constraint

def create_problem_from_mps(filepath: str) -> MIPProblem:
    """
    Reads a file in .mps format and converts it to a MIPProblem object.
    This uses Gurobi as a convenient and robust parser.
    """
    print(f"Reading MPS file: {filepath}...")
    
    # Use a temporary Gurobi environment to read the file.
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        gurobi_model = gp.read(filepath, env=env)
        gurobi_model.update()

    # Translate Gurobi model components to our data classes
    problem_name = gurobi_model.ModelName
    sense = "minimize" if gurobi_model.ModelSense == GRB.MINIMIZE else "maximize"

    problem_variables = [
        Variable(
            name=v.VarName,
            is_integer=(v.VType in [GRB.INTEGER, GRB.BINARY]),
            lb=v.LB,
            ub=v.UB
        ) for v in gurobi_model.getVars()
    ]

    objective_coeffs = {
        obj_expr.getVar(i).VarName: obj_expr.getCoeff(i)
        for i in range((obj_expr := gurobi_model.getObjective()).size())
    }

    problem_constraints = []
    sense_map = {'<': '<=', '>': '>=', '=': '=='}
    for c in gurobi_model.getConstrs():
        row = gurobi_model.getRow(c)
        coeffs = {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())}
        constraint = Constraint(
            coeffs=coeffs,
            sense=sense_map.get(c.Sense, '=='),
            rhs=c.RHS
        )
        problem_constraints.append(constraint)

    print("MPS file parsed successfully.")
    
    return MIPProblem(
        name=problem_name,
        variables=problem_variables,
        objective=objective_coeffs,
        constraints=problem_constraints,
        sense=sense
    )