# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary source to the imaage
COPY ./deployment /app/classifier/deployment
COPY ./src /app/classifier/src

# Chang working directory to deployment
WORKDIR /app/classifier/deployment

# Load pretrained models
RUN python load_pretrained_models.py

# Run the API server
CMD ["python", "start_api.py"]