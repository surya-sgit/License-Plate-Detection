FROM python:3.9-slim

WORKDIR /app

# Copy all necessary files at once
COPY requirements.txt main.py yolo_application.py best_license_plate_model.pt license_plate_detection.ipynb ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "yolo_application.py"]
