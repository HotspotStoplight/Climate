# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Pipenv
RUN pip install pipenv

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock ./

# Install the dependencies from Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Copy the rest of the project files into the container
COPY . .

# Set bash as the default command, allowing interactive access
CMD ["bash"]
