import argparse
import time
from solver.tree_manager import TreeManager
from solver.utilities import setup_logger

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Run the Branch and Bound MIP solver.")
    parser.add_argument("--problem_file", type=str, required=True, help="Path to the MIP problem file (e.g., .mps, .lp).")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the solver configuration file.")

    args = parser.parse_args()

    logger.info(f"Starting MIP solver for problem: {args.problem_file}")
    logger.info(f"Using configuration from: {args.config_file}")

    start_time = time.time()
    try:
        tree_manager = TreeManager(args.problem_file, args.config_file)
        incumbent_solution, incumbent_objective = tree_manager.solve()
    except Exception as e:
        logger.error(f"An error occurred during solver execution: {e}")
        incumbent_solution = None
        incumbent_objective = None

    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\n--- Solver Summary ---")
    if incumbent_solution:
        logger.info(f"Optimal solution found. Objective Value: {incumbent_objective:.4f}")
        logger.info("Solution details (first 10 variables):")
        for i, (var, val) in enumerate(incumbent_solution.items()):
            if i >= 10: break
            logger.info(f"  {var}: {val:.4f}")
    else:
        logger.info("No integer-feasible solution found.")
    logger.info(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
