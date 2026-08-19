import logging
import sys
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        try:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "rag_application.log")
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, just continue with console logging
            pass
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger