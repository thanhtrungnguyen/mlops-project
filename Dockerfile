# Use an official lightweight Python image.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the Poetry configuration files.
COPY pyproject.toml poetry.lock* /app/

# Install Poetry.
RUN pip install --no-cache-dir poetry

# Configure Poetry to install dependencies into the global environment (avoiding virtualenv creation).
ENV POETRY_VIRTUALENVS_CREATE=false

# Install production dependencies (skip dev dependencies).
RUN poetry install --no-dev --no-interaction

# Copy the entire project into the container.
COPY . /app

# Expose the port that the Flask app will listen on.
EXPOSE 5000

# Command to run the Flask application.
CMD ["python", "src/webapp/app.py"]
