import numpy as np
import mlflow
from mlflow.models import set_model


class PreferenceUpdater(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        return np.random.random(len(model_input))


set_model(PreferenceUpdater())
