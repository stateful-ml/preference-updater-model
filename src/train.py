import mlflow
from mlflow.pyfunc import log_model
import model
import dotenv

dotenv.load_dotenv()


def main():
    mlflow.set_experiment("Preference updater approach 1")
    with mlflow.start_run():
        log_model(
            artifact_path="model",
            python_model=model.__file__,
            registered_model_name=model.PreferenceUpdater.__name__.lower(),
        )


if __name__ == "__main__":
    main()
