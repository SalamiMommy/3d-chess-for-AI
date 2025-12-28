"""
Reproduction script for process churn / pickle error stability test.
Uses the new SelfPlayEngine to verify it can survive multiple iterations without crashing.
"""
import logging
import sys
import multiprocessing as mp
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReproChurn")

# Dummy model creation if needed, but we'll try to use real imports
try:
    from training.parallel_self_play import SelfPlayEngine
    from models.graph_transformer import create_optimized_model
except ImportError:
    # Adjust path if running from root
    sys.path.append(str(Path.cwd()))
    from training.parallel_self_play import SelfPlayEngine
    from models.graph_transformer import create_optimized_model

def create_dummy_checkpoint(path):
    logger.info(f"Creating dummy checkpoint at {path}")
    model = create_optimized_model("small")
    torch.save(model.state_dict(), path)

def run_test(iterations=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running stability test on {device} for {iterations} iterations...")
    
    # 1. Setup Engine
    engine = SelfPlayEngine(num_parallel=4, device=device)
    
    # 2. Create dummy checkpoint
    ckpt_path = "repro_dummy_model.pt"
    create_dummy_checkpoint(ckpt_path)
    
    try:
        for i in range(1, iterations + 1):
            logger.info(f"--- Iteration {i} ---")
            
            # A. Update Model (Restarts Server)
            logger.info("Updating model...")
            engine.update_model(ckpt_path, model_size="small")
            
            # B. Generate Games
            # Run just a few games to exercise the queues
            logger.info("Generating games...")
            examples = engine.generate_games(
                num_games=4, 
                opponent_types=["random", "random"],
                max_moves=10  # Short games
            )
            
            logger.info(f"Got {len(examples)} examples.")
            
            if not examples:
                logger.warning("No examples returned!")
                
    except Exception as e:
        logger.error(f"Test failed at iteration {i}: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down...")
        engine.shutdown()
        # Cleanup
        if Path(ckpt_path).exists():
            Path(ckpt_path).unlink()
            
    logger.info("Test PASSED.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_test()
