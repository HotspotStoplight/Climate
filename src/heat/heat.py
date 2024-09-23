import argparse
import os


from src.config.config import HEAT_MODEL_ASSET_ID
from src.constants.constants import HEAT_INPUT_PROPERTIES, HEAT_OUTPUTS_PATH, HEAT_SCALE
from src.heat.utils import make_training_data, train_and_evaluate
from src.utils.utils import (
    make_snake_case,
    predict,
)

from .utils import process_data_to_classify


import ee
from dotenv import load_dotenv
from google.cloud import storage


load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")


ee.Initialize(project=GOOGLE_CLOUD_PROJECT)
storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)


def main(place_name: str) -> None:
    """
    Run the heat prediction pipeline for a given place.

    Args:
        place_name (str): The name of the place to generate heat hazard predictions for.
    """

    print(f"Predicting for {place_name}...")

    snake_case_place_name: str = make_snake_case(place_name)

    make_training_data()

    train_and_evaluate(bucket)

    predict(
        place_name,
        "heat",
        f"predicted_heat_hazard_{snake_case_place_name}",
        bucket,
        HEAT_OUTPUTS_PATH,
        HEAT_SCALE,
        process_data_to_classify,
        HEAT_INPUT_PROPERTIES,
        HEAT_MODEL_ASSET_ID,
    )

    print(f"Prediction for {place_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run heat prediction pipeline for a given place."
    )
    parser.add_argument(
        "place_name", type=str, help="The name of the place to predict on"
    )

    args = parser.parse_args()
    main(args.place_name)
