import mlflow
import pandas as pd
 
logged_model = 'runs:/5c260d1356fa4775b31bac75d69e4c45/model'
 
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
 
# Predict on a Pandas DataFrame.
predict = loaded_model.predict(pd.DataFrame([[6.7, 3.1, 5.6, 2.4]]))
 
print(predict)