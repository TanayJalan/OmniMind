# ─────────────────────────────────────────────────────────────────
#  Dockerfile — OmniMind Streamlit App
#
#  Build:  docker build -t omnimind .
#  Run:    docker run -p 7860:7860 --env-file .env omnimind
#
#  Port 7860 is the default for HuggingFace Spaces — using it
#  locally too keeps things consistent.
# ─────────────────────────────────────────────────────────────────

# Use slim Python image to keep the container small
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# tesseract-ocr  → OCR engine used by pytesseract
# libgl1          → required by OpenCV
# git             → needed by some HuggingFace downloads
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*   # clean up to reduce image size

# Copy and install Python dependencies first (Docker layer caching)
# If requirements.txt doesn't change, this layer is cached and
# subsequent builds are much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Create the data directories the app expects
RUN mkdir -p data/uploads data/chroma_db mlruns

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Health check — Docker will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run Streamlit on port 7860
# --server.address=0.0.0.0 makes it accessible outside the container
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
