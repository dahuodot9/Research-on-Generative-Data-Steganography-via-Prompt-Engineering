# config.py
import os

API_CONFIGS = {
    "doubao": {
        "api_key": "xxxxx",
        "base_url": "xxxxx",
        "model_name": "xxxxx",
    },
    "wenxin": {
        "api_key": "xxxxx",  
        "model_name": "xxxxx",            
    },
    "kimi": {
        "api_key": "xxxxx",
        "base_url": "xxxxx",
        "model_name": "xxxxx",
    },
}

MODELS_TO_TEST = ["wenxin"]  
QUESTIONS_FILE = "data/questions.txt"            
OUTPUT_DIR = "data/raw"                          
NUM_SAMPLES_PER_TEMPLATE = 3                     
TEMPERATURE = 0.8                                

CLASSIFIER_BACKBONE = "bert-base-chinese"      
