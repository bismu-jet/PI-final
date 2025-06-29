from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Node:
    """
    Represents a node in the Branch and Bound tree.
    """
    node_id: int
    parent_id: Optional[int]
    local_constraints: List[Tuple[str, str, float]]  # e.g., [(\'x1\', \'<=\', 0), (\'x2\', \'>=\', 1)]
    lp_objective: Optional[float]
    lp_solution: Optional[Dict[str, float]]
    status: str  # e.g., \'PENDING\', \'SOLVED\', \'PRUNED_INFEASIBLE\', \'FATHOMED\'
