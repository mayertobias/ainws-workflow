FROM apache/airflow:2.8.1-python3.11

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories first
RUN mkdir -p /opt/airflow/plugins /opt/airflow/dags /opt/airflow/config /opt/airflow/shared-models /opt/airflow/shared-data

# Copy custom operators and plugins (if they exist)
COPY --chown=airflow:root plugins/ /opt/airflow/plugins/
COPY --chown=airflow:root dags/ /opt/airflow/dags/

# Copy config files if they exist, otherwise create a placeholder
COPY --chown=airflow:root config/ /opt/airflow/config/

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/plugins"
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True 