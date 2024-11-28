FROM python:3.12-slim

WORKDIR /frontend

COPY requirements.txt .

# Install zlib development headers, libjpeg, and build-essential
RUN apt-get update && apt-get install -y zlib1g-dev libjpeg-dev build-essential

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Set environment variables for Streamlit (optional)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Command to run the Streamlit app
CMD ["streamlit", "run", "myApp.py", "--server.port=8501", "--server.address=0.0.0.0"]