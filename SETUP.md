# Setup Guide for Climate Repository

.
This guide will walk you through the process of setting up the project environment for the Hotspot Stoplight climate repository, which is a monorepo containing each of the various climate modules (currently flood and heat)

## Prerequisites

Before you begin, ensure you have the following tools installed:

1. `pipenv`: Follow the instructions at https://pipenv.pypa.io/en/latest/installation.html
2. `pyenv`: Follow the instructions at https://github.com/pyenv/pyenv

## Setup Steps

To set up the repository, clone the repo, install Python 3.11 as the local version, install the depencies from the Pipfile, and activate the virtual environment (if necessary).

```
git clone https://github.com/HotspotStoplight/Climate
pyenv install 3.11
pyenv local 3.11
pipenv install
pipenv shell
```

## Setting User Credentials for Google Cloud

### Install and Initialize gcloud CLI

1. Create a `credentials/` subdirectory in the root directory. Add it to the `.gitignore` file with `credentials/`.
2. Download a service key from Google Cloud Storage (GCS) and place it in the `credentials/` directory.
3. Create a `.env` file in the `src/` directory with the following content:

```
GOOGLE_CLOUD_PROJECT=your-project-name
GOOGLE_CLOUD_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json
```

4. Ensure that the Google Earth Engine (GEE) API is enabled for your chosen project and that the project is registered for GEE usage.
5. Run `earthengine authenticate` in your terminal.

### Additional Setup for Flood Module

For the flood module:

1. Download the full [EMDAT disasters database](https://public.emdat.be/) as an Excel sheet.
2. Upload it to your GCP storage bucket.
3. Update the `EMDAT_DATA_PATH` variable in the config file to match this path.

## Running the Application

To run the application:

1. Activate the virtual environment:

   ```
   pipenv shell
   ```

2. Run the full module:

   ```
   python -m src.main [Country Name]
   ```

   Or run individual modules:

   ```
   python -m src.heat.heat [Country Name]
   python -m src.flood.flood [Country Name]
   ```

## Python Development

To set up your local Python environment for development outside of Docker:

1. Install the same Python version as specified in the Dockerfile. Use `pyenv` to manage multiple distributions if needed.
2. Use `pipenv` to create a virtual environment.
3. Install the pip dependencies defined in the Pipfile into your virtual environment.
4. Install any additional executables with `apt-get`.

Now you can develop in Python in your terminal and IDE, and run unit tests with `pytest`.
