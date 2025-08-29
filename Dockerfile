# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project to the working directory
COPY . .

# Install Gradio and other dependencies
RUN pip install --no-cache-dir gradio
RUN pip install --no-cache-dir -e .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]
