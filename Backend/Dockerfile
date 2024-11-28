# to run this docker image use the following command: docker run --env-file .env -p 8000:8000 car-fraud-backend
# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /backend

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .



# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]