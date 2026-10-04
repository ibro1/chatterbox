# Use an official Python runtime as a parent image
FROM python:3.11-slim

# git: pyproject pulls resemble-perth from GitHub. ffmpeg: mp3 output from /v1/audio/speech.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# CPU-only torch keeps the image gigabytes smaller on a box without a GPU.
# Build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 for CUDA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url ${TORCH_INDEX}

# Copy the entire project to the working directory
COPY . .

# Chatterbox plus its pinned Gradio; FastAPI and uvicorn serve the UI and the API together
RUN pip install --no-cache-dir -e . fastapi uvicorn

# Mount persistent volumes on these paths (Dokploy: Advanced -> Volumes) so uploaded voices
# and the downloaded model weights survive redeploys:
#   /app/voices                 reference WAVs for /v1/voices (CHATTERBOX_VOICES_DIR)
#   /root/.cache/huggingface    Hugging Face model cache
RUN mkdir -p /app/voices /root/.cache/huggingface

# Expose the port the UI and API share
EXPOSE 7860

# Command to run the server
CMD ["python", "app.py"]
