import requests
import os

url = "http://127.0.0.1:8000/predict"

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the file path to the CSV file
file_path = os.path.join(current_dir, "..", "data", "test_data.csv")

# Optionally, check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, "rb") as f:
    files = {"file": ("test_data.csv", f, "text/csv")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print(f"Error: {response.status_code}")
    print("Message:", response.text)