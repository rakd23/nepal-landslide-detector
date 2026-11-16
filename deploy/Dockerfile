# Nepal Landslide Detector: container for Hugging Face Spaces (or any Docker host).
# The free Spaces hardware has no GPU, so PyTorch is installed in its smaller CPU-only build.
FROM python:3.11-slim

# The rasterio wheel's bundled GDAL needs libexpat, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV HOME=/home/user \
    GEOAI_ROOT=/app/model \
    MAX_QUEUE=3 \
    DEMO_MODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
