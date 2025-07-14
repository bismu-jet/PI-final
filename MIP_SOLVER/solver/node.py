from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass(order=False)
class Node:
    """A node in the Branch and Bound tree."""
    node_id: int
    parent_id: Optional[int]
    lp_objective: Optional[float]
    depth: int = 0
    
    # Static variables to control the global search strategy.
    switch_to_bb: bool = False
    is_maximization: bool = False

    # Branching constraints and LP basis information.
    local_constraints: List[Tuple[str, str, float]] = field(default_factory=list)
    status: str = 'PENDING'
    lp_solution: Optional[Dict[str, float]] = None
    vbasis: Optional[Dict[str, int]] = None
    cbasis: Optional[Dict[str, int]] = None

    def __lt__(self, other: 'Node') -> bool:
        """Custom less-than comparator for the priority queue (min-heap)."""
        # If switched, use Best-Bound search based on LP objective.
        if Node.switch_to_bb:
            # For max problems, explore larger objectives first (reverse for min-heap).
            if Node.is_maximization:
                return self.lp_objective > other.lp_objective
            # For min problems, explore smaller objectives first.
            else:
                return self.lp_objective < other.lp_objective
        
        # Initially, use Depth-First Search by prioritizing deeper nodes.
        else:
            return self.depth > other.depth