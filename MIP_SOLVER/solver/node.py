from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class Node:
    """Represents a node in the Branch and Bound tree."""
    node_id: int
    parent_id: Optional[int]
    local_constraints: List[Tuple[str, str, float]] = field(default_factory=list)
    status: str = 'PENDING'
    lp_objective: Optional[float] = None
    lp_solution: Optional[Dict[str, float]] = None
    # --- NEW: Store the basis information ---
    vbasis: Optional[Dict[str, int]] = None
    cbasis: Optional[Dict[str, int]] = None