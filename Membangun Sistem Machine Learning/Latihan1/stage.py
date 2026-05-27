import mlflow.pyfunc
import pandas as pd
 
# URI model dalam Model Registry
model_uri = "models:/iris_classifier/Staging" # MLflow Old Version < 2.9.0
 
# Load model
model = mlflow.pyfunc.load_model(model_uri)
 
# Gunakan model untuk prediksi
input_data = pd.DataFrame([[6.7, 3.1, 5.6, 2.4]])
predictions = model.predict(input_data)
print(predictions)