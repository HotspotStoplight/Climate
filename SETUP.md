# Setup Guide for Climate Repository

This guide will walk you through the process of setting up the project environment for the Hotspot Stoplight climate repository, which is a monorepo containing various climate modules (currently flood and heat).

## Prerequisites

This project relies heavily on Google Earth Engine, so, before anything else, make sure you have a Google Cloud account set up. Once you do, install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and authenticate.

Next, clone this repository:

```
git clone https://github.com/HotspotStoplight/Climate
```

Create a `/credentials` subdirectory in the root of the repository. This will be ignored by Git. Download a service account key associated with the HotspotStoplight project, place it in the `/credentials` subdirectory, and name it `service-account-key.json`.

1. Create a `credentials/` subdirectory in the root directory and add it to `.gitignore`.
2. Place the downloaded service account key in the `credentials/` directory.
3. Create a `.env` file in the `src/` directory with the following content:

   ```
   GOOGLE_CLOUD_PROJECT=your-project-name
   GOOGLE_CLOUD_BUCKET=your-bucket-name
   GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json
   ```

4. Enable the Google Earth Engine (GEE) API for your project and register for GEE usage.
5. Run `earthengine authenticate` in your terminal.

### Additional Setup for Flood Module

For the flood module:

1. Download the full [EMDAT disasters database](https://public.emdat.be/) as an Excel sheet.
2. Upload it to your GCP storage bucket.
3. Update the `EMDAT_DATA_PATH` variable in the config file to match the path.

## Containerized Application

### Setup

This repository can be run in a Docker container. Assuming you have [Docker installed](https://docs.docker.com/engine/install/), navigate to the root directory of the repository and run:

```
docker-compose build climate
```

### Running the Application

With the container built, run:

```
docker-compose run climate
```

This will open a bash terminal with the pipenv shell activated. You can then run relevant commands as if you were working locally.

To run the full script:

```
python -m src.main [Country Name]
```

Or run individual modules:

```
python -m src.heat.heat [Country Name]
python -m src.flood.flood [Country Name]
```

## Local Development

### Setup

For local development, install the following tools:

1. `pipenv`: Follow the instructions at https://pipenv.pypa.io/en/latest/installation.html
2. `pyenv`: Follow the instructions at https://github.com/pyenv/pyenv

Once installed and the repo is cloned, navigate to the root directory, install Python 3.11, install dependencies, and activate the virtual environment:

```
pyenv install 3.11
pyenv local 3.11
pipenv install
pipenv shell
```

### Running the Application

To run the application outside of Docker:

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
