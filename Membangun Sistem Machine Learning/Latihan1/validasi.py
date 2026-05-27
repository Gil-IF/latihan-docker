from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input    

import mlflow
import pandas as pd
 
logged_model = 'runs:/ba7a97b37ca3420d9ecf281e84f6e952/model'
 
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
 
# Predict on a Pandas DataFrame.
predict = loaded_model.predict(pd.DataFrame([[6.7, 3.1, 5.6, 2.4]]))
 
print(predict)