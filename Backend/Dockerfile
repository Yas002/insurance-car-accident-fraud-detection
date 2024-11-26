# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /backend

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend application code into the container
COPY . .

# (Optional) Add /backend to PYTHONPATH to allow module imports from backend/
ENV PYTHONPATH="${PYTHONPATH}:/backend"

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]