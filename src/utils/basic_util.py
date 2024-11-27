from src.logger.basic_logging import logging
import os
import pickle

def save_object(file_path,obj) -> bool:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error("failed to write object to file", exc_info=e)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        logging.error("failed to write object to file", exc_info=e)