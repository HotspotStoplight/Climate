from data_utils.make_training_data import make_training_data
from data_utils.export_and_monitor import start_export_task
import ee
from google.cloud import storage
import re


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None


def check_and_export_geotiffs_to_bucket(
    bucket_name, fileNamePrefix, flood_dates, bbox, fishnet, num_grids, scale
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    existing_files = list(bucket.list_blobs(prefix=fileNamePrefix))
    existing_dates = [
        extract_date_from_filename(file.name)
        for file in existing_files
        if extract_date_from_filename(file.name) is not None
    ]

    tasks = []

    for index, (start_date, end_date) in enumerate(flood_dates):
        if start_date.strftime("%Y-%m-%d") in existing_dates:
            print(f"Skipping {start_date}: data already exist")
            continue

        training_data_result = make_training_data(bbox, start_date, end_date)
        if training_data_result is None:
            print(
                f"Skipping export for {start_date} to {end_date}: No imagery available."
            )
            continue

        for grid_index in range(num_grids):
            grid_feature = fishnet.getInfo()["features"][grid_index]
            grid_geom = ee.Geometry.Polygon(grid_feature["geometry"]["coordinates"])

            clipped_training_data = training_data_result.clip(grid_geom)

            specificFileNamePrefix = (
                f"{fileNamePrefix}input_data_{start_date}_chunk_{grid_index + 1}"
            )
            export_description = f"input_data_{start_date}_chunk_{grid_index + 1}"

            print(f"Exporting chunk {grid_index + 1} of {num_grids} for {start_date}")
            task = start_export_task(
                clipped_training_data.toShort(),
                export_description,
                bucket_name,
                specificFileNamePrefix,
                scale,
            )
            tasks.append(task)

    return tasks
