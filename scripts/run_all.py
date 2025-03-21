import logging

# Configure logging
def setup_logging():
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set up root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("inventory_management") 