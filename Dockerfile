# Base image - slim keeps the build lean since we don't need the full Python image
FROM python:3.11-slim

# Set a proper working directory instead of dumping everything in root
WORKDIR /app

# Install system deps needed by psycopg2 and matplotlib before copying code
# (doing this first means Docker caches this layer and skips it on future rebuilds
# unless requirements.txt actually changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy just requirements first so pip install gets cached separately from code changes
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# These folders get written to at runtime (data, models, results, logs) -
# create them up front so the container doesn't fail on a missing directory
RUN mkdir -p data/raw data/processed models reports/metrics reports/figures \
    reports/explainability results logs

# Streamlit's default port, exposed in case the dashboard app is run from this image
EXPOSE 8501

# Runs the full pipeline end to end by default.
# Override at `docker run` time if you just want one stage, e.g.:
#   docker run <image> python src/ml/predict_failure.py
CMD ["python", "run_pipeline.py"]