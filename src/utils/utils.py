import os
import re
import time

import ee
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Bucket

from datetime import datetime, timezone
from typing import List, Optional, Any


from src.utils.pygeoboundaries.main import get_area_of_interest

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")


ee.Initialize(project=GOOGLE_CLOUD_PROJECT)
storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)


# function to make the place name snake case-------------------------------------------------------
def make_snake_case(place_name: str) -> str:
    """
    Converts a given place name into snake_case format.

    Args:
        place_name (str): The input string representing a place name.

    Returns:
        str: The place name converted to snake_case (lowercase, spaces replaced with underscores).
    """
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


def monitor_tasks(tasks: List[ee.batch.Task], sleep_interval: int = 10) -> None:
    """
    Monitors the completion status of Earth Engine tasks and prints progress.

    Args:
        tasks (List[ee.batch.Task]): A list of Earth Engine tasks to monitor.
        sleep_interval (int): Time in seconds to wait between status checks (default is 10 seconds).
    """
    print("Monitoring Earth Engine tasks...")
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


def check_and_export_geotiffs_to_bucket(file_name_prefix, dates, bbox, scale=90):
    existing_files = list(bucket.list_blobs(prefix=file_name_prefix))
    existing_dates = [
        extract_date_from_filename(file.name)
        for file in existing_files
        if extract_date_from_filename(file.name) is not None
    ]

    tasks = []

    for index, (start_date, end_date) in enumerate(dates):
        if start_date.strftime("%Y-%m-%d") in existing_dates:
            print(f"Skipping {start_date}: data already exists")
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
            geotiff, export_description, bucket.name, specific_file_name_prefix, scale
        )
        tasks.append(task)

    if tasks:
        print("All exports initiated, monitoring task status...")
        # Reuse the task monitoring function
        monitor_tasks(tasks)
    else:
        print("No exports were initiated.")

    print(f"Finished checking and exporting GeoTIFFs. Processed {len(dates)} events.")


# function to check if a file or files exist before proceeding-------------------------------------------------------
def data_exists(prefix: str) -> bool:
    """
    Checks if any data exists in a Google Cloud Storage bucket under a given prefix.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        prefix (str): The prefix to search for within the bucket.

    Returns:
        bool: True if there are any blobs under the given prefix, False otherwise.
    """
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def list_and_check_gcs_files(prefix: str) -> List[str]:
    """
    Checks if files with a specified prefix exist in a Google Cloud Storage (GCS) bucket
    and returns a list of URLs for those files.

    Args:
        prefix (str): The prefix used to filter files within the GCS bucket.

    Returns:
        List[str]: A list of GCS URLs for files that match the prefix, or an empty list
        if no files are found.
    """
    # List blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Check if any files exist with the specified prefix
    if len(blobs) == 0:
        print(f"No files found with prefix '{prefix}' in bucket '{bucket.name}'.")
        return []

    # List and return all files with the specified prefix
    file_urls = [
        f"gs://{bucket.name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]
    return file_urls


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extracts a date in the format YYYY-MM-DD from a given filename.

    Args:
        filename (str): The filename string from which to extract the date.

    Returns:
        Optional[str]: The extracted date in YYYY-MM-DD format if found, otherwise None.
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    return None


# function for stratified sampling based on land cover classes-------------------------------------------------------


# function to read images in a directory from GCS into an image collection-------------------------------------------------------
def convert_bands_based_on_uri(image: ee.Image, uri: str) -> ee.Image:
    """
    Converts the appropriate bands to integers based on the content of the URI.

    Args:
        image (ee.Image): The Earth Engine image to process.
        uri (str): The URI string to determine which bands to convert.

    Returns:
        ee.Image: The image with the relevant bands converted to integer.
    """
    if "flood" in uri:
        bands_to_convert = ["landcover", "flooded_mask"]
    elif "heat" in uri:
        bands_to_convert = ["landcover"]
    else:
        return image  # No bands to convert

    int_bands = [image.select(band).toInt().rename(band) for band in bands_to_convert]
    return image.addBands(int_bands, overwrite=True)


def read_images_into_collection(uri_list: List[str]) -> ee.ImageCollection:
    """
    Reads images from a list of URIs into an Earth Engine image collection and
    applies band conversion automatically based on the content of the URIs.

    Args:
        uri_list (List[str]): A list of URIs pointing to GeoTIFF images.

    Returns:
        ee.ImageCollection: An Earth Engine image collection with processed images.
    """
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in uri_list]

    def process_image(image, uri):
        return convert_bands_based_on_uri(image, uri)

    image_collection = ee.ImageCollection.fromImages(
        [process_image(img, uri) for img, uri in zip(ee_image_list, uri_list)]
    )

    info = image_collection.size().getInfo()
    print(f"Collection contains {info} images.")

    return image_collection


# function to export a trained classifier-------------------------------------------------------
def export_model_as_ee_asset(
    regressor: Any, description: str, asset_id: str
) -> ee.batch.Task:
    """
    Exports an Earth Engine classifier model (regressor) to an Earth Engine asset.

    Args:
        regressor (Any): The Earth Engine classifier model to be exported.
        description (str): A description for the export task.
        asset_id (str): The Earth Engine asset ID where the model will be saved.

    Returns:
        ee.batch.Task: The Earth Engine task object representing the export task.
    """
    # Export the classifier to an Earth Engine asset
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


def predict(
    place_name: str,
    model_type: str,
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
    if data_exists(f"{base_directory}{predicted_image_filename}"):
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
        classified_image,
        snake_case_place_name,
        model_type,
        bucket,
        base_directory,
        scale,
    )

    # Monitor the export task using the reusable monitoring function
    monitor_tasks([task], sleep_interval=600)


# function to export predictions-------------------------------------------------------
def export_predictions(
    classified_image,
    snake_case_place_name: str,  # Now passed directly from predict
    model_type: str,
    bucket: Bucket,
    directory_name: str,
    scale: float,
) -> object:
    """
    Export the predictions to Google Cloud Storage.

    Args:
        classified_image: The image to be exported.
        snake_case_place_name (str): The snake_case formatted place name for the prediction.
        model_type (str): The type of model, e.g., 'flood', 'heat'.
        bucket (Bucket): The Google Cloud Storage bucket where the predictions will be uploaded.
        directory_name (str): The directory name in the bucket where the predictions will be stored.
        scale (float): The scale to be used for the export.

    Returns:
        object: The export task object.
    """
    predicted_image_filename = f"predicted_{model_type}_risk_{snake_case_place_name}"

    export_description = f"{snake_case_place_name} predicted {model_type} risk"

    full_path = f"{directory_name}{predicted_image_filename}"

    task = start_export_task(
        classified_image,
        export_description,
        bucket.name,
        full_path,
        scale,
    )
    return task


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

    # Check if the blob exists
    if not blob.exists():
        # Create a new CSV file with a header if it doesn't exist
        header = "Date,Model Type,Model Asset ID,Training Countries,Input Properties,Sample Size\n"
        blob.upload_from_string(header, content_type="text/csv")

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
    new_row_csv = ",".join(map(str, new_row)) + "\n"

    # Append the new row to the CSV file in GCS
    # Open the blob and append the new row without downloading the entire file
    previous_data = blob.download_as_text()
    blob.upload_from_string(previous_data + new_row_csv, content_type="text/csv")

    print(
        f"Model tracking info for {model_type} appended to {tracking_csv_path} in GCS."
    )
