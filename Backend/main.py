# type  uvicorn main:app --reload  in cmd to run the app
# to build the docker image type: cd to Backend directory, then type: docker build -t car-fraud-backend .

from clean_data import create_preprocessing_pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from dotenv import load_dotenv
import os






# Load environment variables from the .env file
load_dotenv(".env")

# Set the DagsHub username and token from environment variables
dagshub_username = os.getenv('MLFLOW_TRACKING_USERNAME')
dagshub_token = os.getenv('MLFLOW_TRACKING_PASSWORD')


if dagshub_username and dagshub_token:
    print("Environment variables loaded successfully:")
    print(f"DagsHub Username: {dagshub_username}")
    print(f"DagsHub Token: {dagshub_token}")
else:
    print("Failed to load environment variables.")


# Set the MLflow tracking URI to your existing DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/yassine_msaddak/insurance-car-accident-fraud-detection.mlflow")

# Optionally authenticate with your DagsHub credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# Now you can log your model and experiments directly with MLflow
print(f"Using MLflow to track experiments on DagsHub repo: {dagshub_username}/insurance-car-accident-fraud-detection")


# Function to load the best model based on a specified metric
def load_best_model(experiment_name, metric_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    experiment_id = experiment.experiment_id

    # Escape metric name if it contains spaces or special characters
    order_by_metric = f"metrics.`{metric_name}` DESC"

    # Search for the best run based on the metric
    best_runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=[order_by_metric]
    )

    if not best_runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}' with metric '{metric_name}'.")

    best_run = best_runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id}, {metric_name}: {best_run.data.metrics[metric_name]}")

 # List artifacts for the best run to find the model artifact path
    artifacts = client.list_artifacts(best_run_id)
    model_artifact_path = None

    # Function to recursively search for the model artifact
    def find_model_artifact(run_id, path=''):
        artifacts = client.list_artifacts(run_id, path)
        for artifact in artifacts:
            if artifact.is_dir:
                # Recursively search inside directories
                found = find_model_artifact(run_id, artifact.path)
                if found:
                    return found
            else:
                # Check if the artifact is a model file
                if artifact.path.lower().endswith(('.pkl', '.joblib', '.sav')):
                    return artifact.path
                elif artifact.path.lower().endswith('mlmodel'):
                    # MLflow models have an 'MLmodel' file
                    return path
        return None

    model_artifact_path = find_model_artifact(best_run_id)

    if model_artifact_path is None:
        raise ValueError(f"No model artifact found in run '{best_run_id}'.")

    # Construct the model URI
    if model_artifact_path:
        logged_model = f"runs:/{best_run_id}/{model_artifact_path}"
    else:
        # Model artifact is at the root artifact directory
        logged_model = f"runs:/{best_run_id}/"

    return logged_model


# Example usage
logged_model = load_best_model('insurance-fraud-detection-experiment', 'test_F1_Score')

# Load the model using the logged_model path
model = mlflow.pyfunc.load_model(logged_model)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Car Accident Fraud Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file into a DataFrame
        data = pd.read_csv(file.file)

        # Preprocess the data
        preprocessed_data = create_preprocessing_pipeline(data)

        # Predict using the model
        predictions = model.predict(preprocessed_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))