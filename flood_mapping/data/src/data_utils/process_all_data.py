from datetime import datetime
from data_utils.write_to_cloud import check_and_export_geotiffs_to_bucket
from data_utils.export_and_monitor import start_export_task
from data_utils.monitor_tasks import monitor_tasks
from data_utils.read_from_cloud import read_images_into_collection
from data_utils.new_train_and_eval import train_and_evaluate_classifier
from data_utils.make_data_to_classify import make_non_flooding_data
from data_utils.pygeoboundaries import get_adm_ee
from data_utils.filter_emdat import filter_data_from_gcs
from google.cloud import storage
import ee


cloud_project = "hotspotstoplight"


def process_flood_data(place_name):
    # Check if place_name is a string
    if not isinstance(place_name, str):
        return "Error: Place name must be a string in quotation marks."

    snake_case_place_name = place_name.replace(" ", "_").lower()

    aoi = get_adm_ee(territories=place_name, adm="ADM0")
    bbox = aoi.geometry().bounds()

    date_pairs = filter_data_from_gcs(place_name)

    # Prepare date pairs for processing
    flood_dates = [
        (
            datetime.strptime(start, "%Y-%m-%d").date(),
            datetime.strptime(end, "%Y-%m-%d").date(),
        )
        for start, end in date_pairs
    ]

    # Define Google Cloud Storage bucket name and fileNamePrefix
    bucket_name = "hotspotstoplight_floodmapping"
    directory_name = f"data/{snake_case_place_name}/inputs/"

    # Initialize Google Cloud Storage client and create the new directory
    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(
        directory_name
    )  # This creates a 'directory' by specifying a blob that ends with '/'
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )  # Create the directory

    check_and_export_geotiffs_to_bucket(bucket_name, directory_name, flood_dates, bbox)
    image_collection = read_images_into_collection(bucket_name, directory_name)

    ### use the image collection to train a model------------------------------------------------------------

    print("Training and assessing model...")

    # Capture the entire return value as a single variable
    result = train_and_evaluate_classifier(
        image_collection, bbox, bucket_name, snake_case_place_name
    )
    if result is None:
        print("Training and evaluation failed outright. Exiting...")
        return

    # Unpack the result and check each component
    inputProperties, training = result

    if inputProperties is None:
        print("Error: inputProperties is None.")
    if training is None:
        print("Error: training is None.")

    # If both variables are None, exit the function to avoid further errors
    if inputProperties is None or training is None:
        print("Exiting due to insufficient data for proceeding with classification.")
        return

    # Continue with the rest of your process if both are not None
    print("Both inputProperties and training have valid data.")

    ### make final image to classify probability, write results---------------------------------------------------------------

    final_image = make_non_flooding_data(bbox)

    # Ensure that `training` is not None before proceeding
    if training is None:
        print(
            "Error: Training data is not available for classifier training. Exiting..."
        )
        return

    classifier = (
        ee.Classifier.smileRandomForest(10)
        .setOutputMode("PROBABILITY")
        .train(
            features=training,
            classProperty="flooded_mask",
            inputProperties=inputProperties,
        )
    )
    probabilityImage = final_image.classify(classifier)

    swater = ee.Image("JRC/GSW1_0/GlobalSurfaceWater").select("seasonality")
    swater_mask = swater.gte(10).updateMask(swater.gte(10))
    probabilityImage = probabilityImage.where(swater_mask, 0)

    floodProbFileNamePrefix = f"data/{snake_case_place_name}/outputs/flood_prob"
    tasks = []

    # Start the task for exporting flood probability
    task = start_export_task(
        probabilityImage,
        "Flood probability",
        bucket_name,
        floodProbFileNamePrefix,
        scale=30,
    )
    tasks.append(task)

    # After starting all tasks, monitor them together
    monitor_tasks(tasks, 300)

    print(
        f"Processing for {place_name} completed and data saved to Google Cloud in the {snake_case_place_name} directory."
    )
