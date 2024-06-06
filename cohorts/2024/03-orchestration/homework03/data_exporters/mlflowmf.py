import mlflow
import joblib
import os
import shutil

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    MODEL_NAME = "lr-model"
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run() as run:
        if os.path.exists("/home/model"):
            shutil.rmtree("/home/model")
        mlflow.sklearn.log_model(
        data['lr'],
        "lr_model",
        registered_model_name="linear-regression-model")
        dv = data['dv']
        joblib.dump(dv, "dv.pkl")
        mlflow.log_artifact("dv.pkl")


