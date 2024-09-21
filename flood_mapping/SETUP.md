# Setup Guide for Flood Mapping Repository

Follow these steps to set up the project environment:

## 1. Install `pipenv`
Follow the instructions at https://pipenv.pypa.io/en/latest/installation.html to install `pipenv`. 

## 2. Install pyenv
Follow the instructions at https://github.com/pyenv/pyenv to install pyenv.

## 3. Clone the GitHub Repository
Clone the git repository to your local machine by running the following command in your command line interface (CLI):

`git clone https://github.com/HotspotStoplight/Climate`

## 4. Install a local copy of Python
For this project, you'll need Python 3.9. Navigate to `/Climate/flood_mapping` and run `pyenv install 3.9.1`. You may need to set the local version of Python by also running `pyenv local 3.9.1`.

## 5. Install Dependencies with `pipenv`
In your command line interface (CLI), navigate to the flood_mapping subdirectory with `cd Climate/flood_mapping` and then run `pipenv install`.

## 6. Activate the Virtual Environment
Activate the virtual environment by running:

`pipenv shell`

## Setting User Credentials for Google Cloud

### Install and Initialize gcloud CLI

Follow the instructions to [install the gcloud CLI](https://cloud.google.com/sdk/docs/install). Once the CLI is open, it will ask you to log in. Do so with the appropriate account and pick the relevant cloud project (currently `hotspotstoplight`). Your authentication should automatically be saved to your local machine.

Then run `gcloud auth application-default login` to authenticate.

## Running the Flood Prediction Script
To run the script, navigate to the `flood_mapping` directory and activate the virtual environment with `pipenv shell`. Then, run the main script with. This is done with `python data/src/script.py` follow by a list of countries, e.g.,

```
python3 data/src/script.py Uruguay "El Salvador" Brazil

```
Note that countries composed of multiple words (e.g., United Kingdom) should be enclosed in double quotation marks.

The results of the script (a flood probability map, exposure map, and vulnerability map) will be in the relevant `/outputs` subdirectory for the given country in a Google Cloud bucket.