# main.py

import argparse
import time
import logging

# Import our key components
from solver.tree_manager import TreeManager
from solver.utilities import setup_logger

def main():
    """
    The main entry point for the MIP solver application.
    """
    # --- THIS IS THE SINGLE POINT OF LOGGER CONFIGURATION ---
    # We call it once at the very beginning.
    setup_logger()
    # ----------------------------------------------------

    parser = argparse.ArgumentParser(description="A modular MIP solver using Branch and Bound.")
    parser.add_argument('--problem_file', type=str, required=True, help='Path to the MIP problem file (.mps, .lp)')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file (config.yaml)')
    args = parser.parse_args()

    try:
        logging.info(f"--- Starting MIP Solver ---")
        logging.info(f"Problem file: {args.problem_file}")
        logging.info(f"Configuration file: {args.config_file}")

        # Instantiate the main solver engine
        manager = TreeManager(args.problem_file, args.config_file)
        
        # Start the timer and run the solver
        start_time = time.time()
        solution, objective = manager.solve()
        end_time = time.time()
        
        # --- Final Summary ---
        logging.info("--- Solver Summary ---")
        if solution:
            logging.info(f"Optimal solution found!")
            logging.info(f"Objective Value: {objective:.4f}")
            # Optionally print the first few variables of the solution
            first_n = 10
            logging.info(f"Solution details (first {first_n} variables):")
            for i, (var, val) in enumerate(solution.items()):
                if i >= first_n:
                    break
                if abs(val) > 1e-9: # Only print non-zero variables
                    logging.info(f"  {var}: {val:.4f}")
        else:
            logging.info("No integer-feasible solution was found.")

        logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")
        logging.info(f"--- MIP Solver Finished ---")

    except Exception as e:
        logging.error("An unexpected error occurred during the solve process.", exc_info=True)

if __name__ == "__main__":
    main()