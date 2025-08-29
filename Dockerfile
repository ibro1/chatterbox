# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project to the working directory
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -e .

# (Optional) You can add a command to run a specific script upon container start
# For example, to run the example TTS script:
# CMD ["python", "example_tts.py"]
