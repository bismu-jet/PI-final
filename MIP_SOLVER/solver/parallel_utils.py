# solver/parallel_utils.py
import multiprocessing as mp
from typing import List, Dict

# We'll need the Constraint class later for type hinting.
# This assumes your Constraint class is hashable, which it should be.
from .problem import Constraint 

class SharedState:
    """
    Manages the global state shared across all worker processes in a thread-safe manner.
    
    This class uses multiprocessing primitives to ensure that multiple processes can
    read and write data without causing race conditions or data corruption.
    """
    def __init__(self, num_workers: int, sense: str = "minimize"):
        """
        Initializes the shared state.

        Args:
            num_workers (int): The total number of worker processes.
            sense (str): The optimization sense ("minimize" or "maximize").
        """
        self.sense = sense
        
        # A Manager creates a server process that holds Python objects and allows
        # other processes to manipulate them using proxies. This is essential for
        # sharing complex objects like lists and dictionaries.
        manager = mp.Manager()
        
        # Determine the initial "worst-possible" value for the objective.
        initial_primal = float('inf') if self.sense == 'minimize' else -float('inf')
        
        # --- Shared Data Primitives ---
        
        # 'd' for double-precision float. Stores the best objective value found so far.
        self.best_cost = mp.Value('d', initial_primal)
        
        # A shared dictionary (proxy object) to store the variable assignments of the best solution.
        self.best_solution = manager.dict()
        
        # 'b' for boolean. A simple flag to quickly check if any solution has been found yet.
        self.has_solution = mp.Value('b', False)
        
        # A shared list (proxy object) to store all generated cutting planes.
        self.cut_pool = manager.list()
        
        # 'i' for integer. A counter for how many workers are currently idle.
        self.idle_workers = mp.Value('i', 0)
        
        # A counter for the total number of nodes processed across all workers.
        self.nodes_processed = mp.Value('i', 0)

        # A Lock is a synchronization primitive that ensures only one process
        # can execute a block of code at a time. This is critical for preventing
        # race conditions when updating shared state.
        self.lock = mp.Lock()
        
        # A separate lock for the cut pool allows workers to add cuts without
        # blocking other workers from updating the best solution, and vice-versa.
        self.cut_lock = mp.Lock()

    def update_best_solution(self, cost: float, solution: Dict[str, float]) -> bool:
        """
        Atomically checks if a new solution is better than the current best and updates it if so.

        Returns:
            bool: True if the new solution was accepted as the new best, False otherwise.
        """
        # The 'with self.lock:' block automatically acquires the lock before
        # entering the block and releases it upon exiting. This is the safest way
        # to use locks.
        with self.lock:
            is_better = False
            if self.sense == 'minimize':
                if cost < self.best_cost.value:
                    is_better = True
            else: # maximize
                if cost > self.best_cost.value:
                    is_better = True
            
            if is_better:
                print(f"[SharedState]: New incumbent found! Objective: {cost:.4f}")
                self.best_cost.value = cost
                
                # We must clear and update the shared dictionary, not reassign it.
                self.best_solution.clear()
                self.best_solution.update(solution)
                
                if not self.has_solution.value:
                    self.has_solution.value = True
                
                return True
        return False

    def get_best_cost(self) -> float:
        """Returns the current best objective value."""
        return self.best_cost.value

    def add_cuts(self, new_cuts: List[Constraint]):
        """Adds a list of new, unique cuts to the global pool."""
        with self.cut_lock:
            # To efficiently check for uniqueness, we convert the existing
            # pool to a set. This requires your Constraint class to be hashable.
            existing_cuts = set(self.cut_pool)
            for cut in new_cuts:
                if cut not in existing_cuts:
                    self.cut_pool.append(cut)

    def get_cuts(self) -> List[Constraint]:
        """Returns a copy of the current global cut pool."""
        return list(self.cut_pool)

    def increment_idle_worker_count(self):
        """Atomically increments the count of idle workers."""
        with self.lock:
            self.idle_workers.value += 1

    def decrement_idle_worker_count(self):
        """Atomically decrements the count of idle workers."""
        with self.lock:
            self.idle_workers.value -= 1
            
    def get_idle_worker_count(self) -> int:
        """Returns the current number of idle workers."""
        return self.idle_workers.value

    def increment_nodes_processed(self):
        """Atomically increments the total number of nodes processed."""
        with self.lock:
            self.nodes_processed.value += 1