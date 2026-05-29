from pathlib import Path
import logging


def setup_logging():
    
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)

    #log path to save the log
    LOG_PATH = LOG_DIR / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s  - %(filename)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force = True #since basic config runs only once so using force is true we can overwrite many
    )

