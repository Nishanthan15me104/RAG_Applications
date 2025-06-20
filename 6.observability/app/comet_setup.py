
from comet_ml import Experiment
import logging

import os
from dotenv import load_dotenv

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-observability")

experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="rag-observability",
    workspace="nishanthan-sureshkumar",
    auto_param_logging=False,
    auto_metric_logging=False
)

def log_metric(name, value):
    logger.info(f"{name}: {value}")
    experiment.log_metric(name, value)
