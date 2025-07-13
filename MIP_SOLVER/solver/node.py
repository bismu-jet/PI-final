from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass(order=False)
class Node:
    """
    Represents a node in the Branch and Bound tree.

    The `order=False` is important because we are providing a custom __lt__ method.
    """
    node_id: int
    parent_id: Optional[int]
    lp_objective: Optional[float]
    
    # --- NEW: Fields for Dynamic Search Strategy ---
    depth: int = 0
    
    # --- Static variables to control search strategy ---
    # This is a simple way to communicate the global search strategy to all nodes
    # without passing extra parameters around.
    switch_to_bb: bool = False
    is_maximization: bool = False

    # --- Existing fields ---
    local_constraints: List[Tuple[str, str, float]] = field(default_factory=list)
    status: str = 'PENDING'
    lp_solution: Optional[Dict[str, float]] = None
    vbasis: Optional[Dict[str, int]] = None
    cbasis: Optional[Dict[str, int]] = None

    def __lt__(self, other: 'Node') -> bool:
        """
        Custom less-than comparator for use in a priority queue (heapq).
        This method enables the dynamic search strategy.
        """
        # If the 'switch_to_bb' flag is set, we use Best-Bound search.
        # The priority is determined by the LP objective value.
        if Node.switch_to_bb:
            # heapq is a min-heap. For maximization, we want to explore nodes
            # with larger objectives first, so we reverse the comparison.
            if Node.is_maximization:
                return self.lp_objective > other.lp_objective
            # For minimization, we want smaller objectives, which is the
            # default behavior of a min-heap.
            else:
                return self.lp_objective < other.lp_objective
        
        # Before the switch, we use Depth-First Search (DFS).
        # The priority is given to the deepest node in the tree.
        else:
            return self.depth > other.depth

