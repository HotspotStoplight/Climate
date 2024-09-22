# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container (this is the project root)
WORKDIR /usr/src/app

# Install Pipenv
RUN pip install pipenv

# Copy the Pipfile and Pipfile.lock from the project root into the container
COPY Pipfile Pipfile.lock ./

# Install the dependencies from Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Set bash as the default command, while activating the virtual environment using pipenv
CMD ["pipenv", "run", "bash"]
