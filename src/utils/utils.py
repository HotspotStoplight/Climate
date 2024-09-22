import os
import re
import time

import ee
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Bucket

import csv
from datetime import datetime, timezone
from typing import List, Optional

from src.utils.pygeoboundaries.main import get_area_of_interest

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")


ee.Initialize(project=GOOGLE_CLOUD_PROJECT)


# function to initialize google cloud storage connection-------------------------------------------------------
def initialize_storage_client(project: Optional[str], GOOGLE_CLOUD_BUCKET: Optional[str]) -> Bucket:
    """
    Initialize the Google Cloud Storage client and return the storage bucket.

    Args:
        project (Optional[str]): The Google Cloud project ID.
        GOOGLE_CLOUD_BUCKET (Optional[str]): The name of the Google Cloud Storage bucket.

    Returns:
        Bucket: A Google Cloud Storage bucket object.
    """
    if project is None or GOOGLE_CLOUD_BUCKET is None:
        raise ValueError("Project ID and bucket name must be provided and cannot be None.")
    
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)
    return bucket




bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)


# function to make the place name snake case-------------------------------------------------------
def make_snake_case(place_name):
    return place_name.replace(" ", "_").lower()


# functions to start and monitor ee export tasks-------------------------------------------------------


def start_export_task(geotiff, description, bucket, file_name_prefix, scale):
    print(f"Starting export: {description}")
    task = ee.batch.Export.image.toCloudStorage(
        image=geotiff,
        description=description,
        bucket=bucket,
        fileNamePrefix=file_name_prefix,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def monitor_tasks(tasks, sleep_interval=10):
    """
    Monitors the completion status of provided Earth Engine tasks.

    Parameters:
    - tasks: A list of Earth Engine tasks to monitor.
    - sleep_interval: Time in seconds to wait between status checks (default is 10 seconds).
    """
    print("Monitoring tasks...")
    completed_tasks = set()
    while len(completed_tasks) < len(tasks):
        for task in tasks:
            if task.id in completed_tasks:
                continue

            try:
                status = task.status()
                state = status.get("state")

                if state in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if state == "COMPLETED":
                        print(f"Task {task.id} completed successfully.")
                    elif state == "FAILED":
                        print(
                            f"Task {task.id} failed with error: {status.get('error_message', 'No error message provided.')}"
                        )
                    elif state == "CANCELLED":
                        print(f"Task {task.id} was cancelled.")
                    completed_tasks.add(task.id)
                else:
                    print(f"Task {task.id} is {state}.")
            except ee.EEException as e:
                print(f"Error checking status of task {task.id}: {e}. Will retry...")
            except Exception as general_error:
                print(f"Unexpected error: {general_error}. Will retry...")

        # Wait before the next status check to limit API requests and give time for tasks to progress
        time.sleep(sleep_interval)

    print("All tasks have been processed.")


def check_and_export_geotiffs_to_bucket(
    bucket_name, file_name_prefix, dates, bbox, scale=90
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    existing_files = list(bucket.list_blobs(prefix=file_name_prefix))
    existing_dates = [
        extract_date_from_filename(file.name)
        for file in existing_files
        if extract_date_from_filename(file.name) is not None
    ]

    tasks = []

    for index, (start_date, end_date) in enumerate(dates):
        if start_date.strftime("%Y-%m-%d") in existing_dates:
            print(f"Skipping {start_date}: data already exist")
            continue

        training_data_result = make_training_data(bbox, start_date, end_date)
        if training_data_result is None:
            print(
                f"Skipping export for {start_date} to {end_date}: No imagery available."
            )
            continue

        geotiff = training_data_result.toShort()
        specific_file_name_prefix = f"{file_name_prefix}input_data_{start_date}"
        export_description = f"input_data_{start_date}"

        print(
            f"Initiating export for GeoTIFF {index + 1} of {len(dates)}: {export_description}"
        )
        task = start_export_task(
            geotiff, export_description, bucket_name, specific_file_name_prefix, scale
        )
        tasks.append(task)

    if tasks:
        print("All exports initiated, monitoring task status...")
        monitor_tasks(tasks)
    else:
        print("No exports were initiated.")

    print(f"Finished checking and exporting GeoTIFFs. Processed {len(dates)} events.")


# function to check if a file or files exist before proceeding-------------------------------------------------------
def data_exists(bucket_name, prefix):
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def list_and_check_gcs_files(bucket_name, prefix):
    """Check if files exist in a GCS bucket folder and list them if they do."""
    # Create a GCS client
    client = storage.Client()

    # Obtain the bucket object
    bucket = client.bucket(bucket_name)

    # List blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Check if any files exist with the specified prefix
    if len(blobs) == 0:
        print(f"No files found with prefix '{prefix}' in bucket '{bucket_name}'.")
        return []

    # List and return all files with the specified prefix
    file_urls = [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]
    return file_urls


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None


# function for stratified sampling based on land cover classes-------------------------------------------------------


# function to read images in a directory from GCS into an image collection-------------------------------------------------------
def convert_heat_bands_to_int(image):
    landcover_int = image.select("landcover").toInt()
    return image.addBands(landcover_int.rename("landcover"), overwrite=True)


def convert_flood_bands_to_int(image):
    """Convert the 'landcover' and 'flooded_mask' bands to integers."""
    landcover_int = image.select("landcover").toInt()
    flooded_mask_int = image.select("flooded_mask").toInt()

    return image.addBands(
        [
            landcover_int.rename("landcover"),
            flooded_mask_int.rename("flooded_mask"),
        ],
        overwrite=True,
    )


def read_images_into_collection(uri_list):
    """Read images from a list of URIs into an Earth Engine image collection."""
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in uri_list]
    image_collection = ee.ImageCollection.fromImages(ee_image_list)

    if any("flood" in uri for uri in uri_list):
        image_collection = image_collection.map(convert_flood_bands_to_int)

    if any("heat" in uri for uri in uri_list):
        image_collection = image_collection.map(convert_heat_bands_to_int)

    info = image_collection.size().getInfo()
    print(f"Collection contains {info} images.")

    return image_collection


# function to export a trained classifier-------------------------------------------------------
def export_model_as_ee_asset(regressor, description, asset_id):
    # Export the classifier
    task = ee.batch.Export.classifier.toAsset(
        classifier=regressor,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"Exporting trained {description} with GEE ID {asset_id}.")
    return task


# function to import a trained classifier and classify an image-------------------------------------------------------
def classify_image(
    image_to_classify: ee.Image, input_properties: list[str], model_asset_id: str
) -> ee.Image:
    """
    Classify the image using a pre-trained model.

    Args:
        image_to_classify (ee.Image): The image to be classified.
        input_properties (list[str]): A list of input properties (bands) used for classification.
        model_asset_id (str): The asset ID of the pre-trained model to use for classification.

    Returns:
        ee.Image: The classified image.
    """
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(input_properties).classify(regressor)


# function to make predcitions-------------------------------------------------------
def predict(
    place_name: str,
    model_type: str,  # Added to specify the type of model (e.g., 'flood', 'heat')
    predicted_image_filename: str,
    bucket: Bucket,
    directory_name: str,
    scale: float,
    process_data_to_classify,
    input_properties: List[str],
    model_asset_id: str,
) -> None:
    """
    Main function to predict risk for a given place and export the result.

    Args:
        place_name (str): The name of the place for the prediction.
        model_type (str): The type of model, e.g., 'flood', 'heat'.
        predicted_image_filename (str): The filename for the exported prediction.
        bucket (Bucket): The Google Cloud Storage bucket where the predictions will be uploaded.
        directory_name (str): The directory name in the bucket where the predictions will be stored.
        scale (float): The scale to be used for the export.
        process_data_to_classify: The function to prepare data for classification.
        input_properties (List[str]): A list of input properties (bands) used for classification.
        model_asset_id (str): The asset ID of the pre-trained model to use for classification.
    """
    snake_case_place_name = make_snake_case(place_name)
    base_directory = f"{directory_name}{snake_case_place_name}/"

    # Check if predictions data already exists
    if data_exists(bucket.name, f"{base_directory}{predicted_image_filename}"):
        print(f"Predictions data already exists for {place_name}. Skipping prediction.")
        return

    print("Processing data to classify...")
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, input_properties, model_asset_id
    )

    # Export predictions
    task = export_predictions(
        classified_image, place_name, model_type, bucket, base_directory, scale
    )

    # Monitor the export task
    monitor_tasks([task], 600)


# function to export predictions-------------------------------------------------------
def export_predictions(
    classified_image,
    place_name: str,
    model_type: str,  # Added to specify the type of model (e.g., 'flood', 'heat')
    bucket: Bucket,
    directory_name: str,
    scale: float,
) -> object:
    """
    Export the predictions to Google Cloud Storage.

    Args:
        classified_image: The image to be exported.
        place_name (str): The name of the place for which predictions are being exported.
        model_type (str): The type of model, e.g., 'flood', 'heat'.
        bucket (Bucket): The Google Cloud Storage bucket where the predictions will be uploaded.
        directory_name (str): The directory name in the bucket where the predictions will be stored.
        scale (float): The scale to be used for the export.

    Returns:
        object: The export task object.
    """
    snake_case_place_name: str = make_snake_case(place_name)
    predicted_image_filename: str = (
        f"predicted_{model_type}_risk_{snake_case_place_name}"
    )

    # Adjusted to use `model_type` in the description
    export_description = f"{place_name} predicted {model_type} risk"

    # Corrected to include the filename in the directory path
    full_path = f"{directory_name}{predicted_image_filename}"

    task = start_export_task(
        classified_image,
        export_description,
        bucket.name,
        full_path,
        scale,
    )
    return task

from typing import List
from google.cloud import storage
from datetime import datetime, timezone
import csv

def append_model_tracking_info_to_csv(
    model_asset_id: str,
    training_countries: List[str],
    input_properties: List[str],
    sample_size: int,
    model_type: str,
    bucket: storage.Bucket,
    tracking_csv_path: str,
) -> None:
    """
    Appends model training information to a CSV file in GCS for tracking purposes.
    This function omits sample distribution and evaluation results, which will be tracked later.

    Args:
        model_asset_id (str): The ID of the model asset.
        training_countries (List[str]): The countries used to generate training data.
        input_properties (List[str]): The properties used as model inputs.
        sample_size (int): The size of the training sample.
        model_type (str): The type of model (e.g., 'flood', 'heat', etc.).
        bucket (storage.Bucket): The GCS bucket object to interact with.
        tracking_csv_path (str): The path where the CSV will be saved.

    Returns:
        None
    """
    blob = bucket.blob(tracking_csv_path)

    # Download current CSV if it exists, otherwise create a new one.
    try:
        csv_data = blob.download_as_text()
        rows = list(csv.reader(csv_data.splitlines()))
    except Exception:
        # If the CSV doesn't exist yet, create a header.
        rows = [
            [
                "Date",
                "Model Type",
                "Model Asset ID",
                "Training Countries",
                "Input Properties",
                "Sample Size",
            ]
        ]

    # Prepare the new row with training details
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    new_row = [
        current_date,
        model_type,
        model_asset_id,
        "; ".join(training_countries),
        "; ".join(input_properties),
        str(sample_size),  # Convert sample_size to a string
    ]

    rows.append(list(map(str, new_row)))  # Ensure all items in new_row are strings

    # Upload updated CSV
    csv_output = "\n".join([",".join(map(str, row)) for row in rows])
    blob.upload_from_string(csv_output, content_type="text/csv")

    print(
        f"Model tracking info for {model_type} appended to {tracking_csv_path} in GCS."
    )
