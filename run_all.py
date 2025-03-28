import os
import subprocess
import time
from datetime import datetime
import logging
from pathlib import Path

def setup_logging(log_dir):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the inventory management pipeline."""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('output') / 'execution_logs' / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # Run the main pipeline
        logger.info("Starting inventory management pipeline...")
        
        # Import and run the pipeline
        from src.pipeline import InventoryPipeline
        from src.utils.config import Config
        
        # Load configuration
        config = Config()
        
        # Initialize pipeline
        pipeline = InventoryPipeline(config)
        
        # Run pipeline with existing data file
        results = pipeline.run('data/retail_store_inventory.csv')
        
        logger.info("Pipeline completed successfully!")
        
        # Start the dashboard if available
        if 'dashboard' in results:
            logger.info(f"Starting dashboard server at http://localhost:{config.dashboard.port}")
            results['dashboard'].run(host="0.0.0.0", port=config.dashboard.port, debug=config.dashboard.debug)
        else:
            logger.warning("No dashboard available in pipeline results")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 