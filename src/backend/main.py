import pandas as pd
import h2o
import mlflow
import mlflow.h2o
import io
import sys
import os
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from fastapi import FastAPI, File 
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

# Set up an H2O cluster
h2o.init()
    
# Spin up Mlflow client
client = MlflowClient()

# Get the best model in the experiment used for training by first
# getting all experiments and searching for run with highest auc metric
experiments = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
experiment_ids = [exp.experiment_id for exp in experiments]
runs_df = mlflow.search_runs(
    experiment_ids=experiment_ids,  # List of experiment IDs to search
    run_view_type=ViewType.ACTIVE_ONLY, # View all runs
    order_by=["metrics.best_model_validation_accuracy DESC"],  # Metrics to sort by and sort order
    max_results=1  # Maximum number of runs to return
)

# Extract the run_id and experiment_id of the top run
best_model = None
best_model_uri = None

# First, try to load the most recent model from the H2O models directory
models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "h2o_automl_models")
try:
    # List all model files and get the most recent one
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
        if model_files:
            # Sort to get the most recent model (assuming naming convention)
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
            print(f"Loading model from: {model_path}")
            best_model = h2o.load_model(model_path)
            best_model_uri = model_path
            print(f"Successfully loaded model: {latest_model}")
except Exception as e:
    print(f"Error loading model from h2o_automl_models: {e}")

# If no model found in h2o_automl_models, try MLflow
if best_model is None and not runs_df.empty:
    print("Attempting to load from MLflow...")
    run_id = runs_df.iloc[0]["run_id"]
    experiment_id = runs_df.iloc[0]["experiment_id"]

    # Get the run object to extract artifact info
    artifacts = client.list_artifacts(run_id, path="")

    # Find the H2O model artifact (will have descriptive names like GBM_*, AutoML_*, etc.)
    model_artifact = None
    for artifact in artifacts:
        if artifact.is_dir:
            model_artifact = artifact.path
            break

    # Construct the absolute filesystem path to the model
    mlruns_base = os.path.join(os.path.dirname(__file__), "..", "..", "mlruns")
    if model_artifact:
        model_dir = os.path.join(mlruns_base, str(experiment_id), run_id, "artifacts", model_artifact)
    else:
        model_dir = os.path.join(mlruns_base, str(experiment_id), run_id, "artifacts")
    
    # Normalize path for Windows compatibility
    model_dir = os.path.normpath(model_dir)

    # Load best model directly from filesystem using h2o
    try:
        print(f"Attempting to load model from: {model_dir}")
        best_model = h2o.load_model(model_dir)
        best_model_uri = model_dir
        print(f"Successfully loaded model from: {model_dir}")
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")

if best_model is None:
    print("WARNING: Model not found. Train a model first using: python src/backend/train.py")

# Create a FastAPI app
app = FastAPI()

# Define the endpoint for prediction
@app.post("/predict")
async def predict(file: bytes = File(...)):
    
    if best_model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please train the model first using: python src/backend/train.py"}
        )

    # Read the uploaded CSV file in bytes format and convert to pandas dataframe
    data = io.BytesIO(file)
    test_df = load_data(data)
    test_df = remove_id_column(test_df)
    test_df = h2o.H2OFrame(test_df)

    test_df = match_test_set_types(test_df, "references/train_processed_column_types.json")

    print(test_df)

    # Generate predictions using the best model
    preds = best_model.predict(test_df)
    preds = preds.as_data_frame()['predict']

    # Convert predictions to json file
    json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
