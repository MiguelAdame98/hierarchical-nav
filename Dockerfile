# Use the Huawei Ascend PyTorch image as the base image
FROM ubuntu:22.04
# Create a new user
RUN apt update && apt install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m 2025_06

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Set proper permissions so the non-root user can access files
RUN chown -R 2025_06:2025_06 /app

# Switch to root user to install dependencies
USER root

# Verify copied files
RUN ls -l /app

# Install local package
RUN pip install  /app/gym-minigrid_minimal-1

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    matplotlib==3.5.3 \
    h5py==3.8.0 \
    torch==2.3.1 \
    PyYAML==6.0.1 \
    tqdm==4.66.4 \
    imageio==2.34.1 \
    pyquaternion==0.9.9 \
    dill==0.3.7 \
    gym==0.17.0 \
    imageio-ffmpeg==0.5.1 \
  && pip install --upgrade imageio imageio-ffmpeg

# Switch back to the non-root user for safety
USER 2025_06

# Default command (replace with actual entry script if needed)
CMD ["/bin/bash"]

