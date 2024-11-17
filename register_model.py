from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

run_id = "3fd65e2f697e40078e8e83adf3ee4ce9"

model_path = "file:///workspaces/codespaces-blank/mlruns/526878055837955901/3fd65e2f697e40078e8e83adf3ee4ce9/artifacts/Random Forest"

model_uri = f"runs:/{run_id}/{model_path}"

model_name = "diabetes-rf"

result = mlflow.register_model(model_uri,model_name)

import time
time.sleep(5)

# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="day",
    value="sat"
)