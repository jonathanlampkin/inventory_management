"""Main entry point for inventory management system."""

import argparse
import logging
from pathlib import Path

from .pipeline import InventoryPipeline
from .utils.config import Config
from .utils.logging_config import setup_logging

def main():
    """Run the inventory management pipeline."""
    parser = argparse.ArgumentParser(description="Inventory Management System")
    parser.add_argument("--config", default="config/config.json", help="Path to config file")
    parser.add_argument("--data", required=True, help="Path to input data")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    logger.info("Starting Inventory Management System")

    try:
        # Load configuration
        config = Config.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Initialize and run pipeline
        pipeline = InventoryPipeline(config)
        results = pipeline.run(args.data)
        logger.info("Pipeline execution completed successfully")

        # Run dashboard
        from .visualization.dashboard import Dashboard
        dashboard = Dashboard(results)
        dashboard.run()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 