import csv
import json
import logging
import os
import yaml


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

logger = get_logger()


def check_path_exists(path: str) -> bool:
    return os.path.exists(path)


def read_yaml_file(yaml_location: str) -> dict:
    with open(yaml_location, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)


def read_csv_file(csv_location:str, skip_header:bool=True) -> list[str]:
    with open(csv_location, newline='', encoding='unicode_escape') as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader)  # Skip the header
        data = [str(row[0]) for row in reader]
    return data


def write_json_file(data: dict, output_path: str) -> None:
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

        logger.info(f"Results have been written to {output_path}")
