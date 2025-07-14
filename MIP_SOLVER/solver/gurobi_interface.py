import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Any, Optional

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()

def solve_lp_relaxation(problem: MIPProblem, 
                        local_constraints: List[Tuple[str, str, float]], 
                        cuts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Solves the LP relaxation of the MIP problem with its original objective.
    """
    result = {'status': 'UNKNOWN'}
    relaxed_model = None

    try:
        relaxed_model = problem.model.relax()
        
        relaxed_model.setParam(GRB.Param.Presolve, 0)
        relaxed_model.setParam(GRB.Param.DualReductions, 0) 
        
        for var_name, sense, value in local_constraints:
            var = relaxed_model.getVarByName(var_name)
            if var is not None:
                if sense == '<=': relaxed_model.addConstr(var <= value)
                else: relaxed_model.addConstr(var >= value)
        
        if cuts:
            for i, cut in enumerate(cuts):
                
                # --- NOVO LOG DE DEPURAÇÃO ---
                # Imprime o corte exato que estamos tentando adicionar para diagnóstico.
                logger.debug(f"Adicionando o corte ao modelo: {cut}")
                # --- FIM DO LOG ---

                cut_coeffs_dict = cut['coeffs']
                cut_rhs = cut['rhs']
                cut_sense = cut['sense']
                
                new_expr = gp.LinExpr()
                for var_name, coeff in cut_coeffs_dict.items():
                    var = relaxed_model.getVarByName(var_name)
                    if var is not None:
                        new_expr.add(var, coeff)
                
                if cut_sense == GRB.GREATER_EQUAL:
                    relaxed_model.addConstr(new_expr >= cut_rhs, name=f"cut_{i}")
                else:
                    relaxed_model.addConstr(new_expr <= cut_rhs, name=f"cut_{i}")

        relaxed_model.optimize()

        if relaxed_model.Status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
            result['objective'] = relaxed_model.ObjVal
            result['solution'] = {v.VarName: v.X for v in relaxed_model.getVars()}
            result['vbasis'] = {v.VarName: v.VBasis for v in relaxed_model.getVars()}
            result['cbasis'] = {c.ConstrName: c.CBasis for c in relaxed_model.getConstrs()}
        else:
            result['status'] = 'INFEASIBLE' if relaxed_model.Status == GRB.INFEASIBLE else 'UNBOUNDED'

    except gp.GurobiError as e:
        logger.error(f"Um erro ocorreu no Gurobi durante a resolução da relaxação LP: {e}", exc_info=True)
        result['status'] = 'ERROR'
    
    finally:
        if relaxed_model:
            relaxed_model.dispose()

    return result

def solve_lp_with_custom_objective(problem: MIPProblem,
                                   objective_coeffs: Dict[str, float]) -> Dict[str, Any]:
    result = {'status': 'UNKNOWN'}
    model_copy = None
    try:
        model_copy = problem.model.relax()
        objective_expr = gp.LinExpr()
        for var_name, coeff in objective_coeffs.items():
            var = model_copy.getVarByName(var_name)
            if var is not None:
                objective_expr.add(var, coeff)
        
        model_copy.setObjective(objective_expr, GRB.MINIMIZE)
        model_copy.optimize()

        if model_copy.Status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
            result['objective'] = model_copy.ObjVal
            result['solution'] = {v.VarName: v.X for v in model_copy.getVars()}
        else:
            result['status'] = 'INFEASIBLE'
    except gp.GurobiError as e:
        logger.error(f"A Gurobi error occurred during custom objective LP solve: {e}")
        result['status'] = 'ERROR'
    finally:
        if model_copy:
            model_copy.dispose()
            
    return result

def solve_sub_mip(problem: MIPProblem,
                  fixed_vars: Dict[str, float],
                  time_limit: float) -> Dict[str, Any]:
    result = {'status': 'UNKNOWN'}
    sub_mip_model = None
    try:
        sub_mip_model = problem.model.copy()
        sub_mip_model.setParam('Presolve', 0)
        sub_mip_model.setParam('Cuts', 0)
        sub_mip_model.setParam('Heuristics', 0)
        sub_mip_model.setParam('TimeLimit', time_limit)
        for var_name, value in fixed_vars.items():
            var = sub_mip_model.getVarByName(var_name)
            if var is not None:
                var.LB, var.UB = value, value
        sub_mip_model.optimize()
        if sub_mip_model.SolCount > 0:
            result['status'] = 'FEASIBLE'
            result['objective'] = sub_mip_model.ObjVal
            result['solution'] = {v.VarName: v.X for v in sub_mip_model.getVars()}
        else:
            result['status'] = 'NO_SOLUTION_FOUND'
    except gp.GurobiError as e:
        logger.error(f"A Gurobi error occurred during sub-MIP solve: {e}")
        result['status'] = 'ERROR'
    finally:
        if sub_mip_model:
            sub_mip_model.dispose()
            
    return result