# NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Copy the requirements file
COPY requirements.txt .

# Install the requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Working directory
WORKDIR /workspace

