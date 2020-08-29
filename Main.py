# importing necessary libraries

import Pandas as pd
from Data import split_data
from Model import logisticregressor,xgbclassifier,multilayerperceptron,RFClassifier,ROC_curve

data=pd.read_csv('bank_data.csv',sep=';')
X_train,y_train,X_test,y_test=split_data(data)

#First model
logisticregressor(X_train,y_train,X_test,y_test)
# ROC curve for the model
ROC_curve(X_test,y_test,model_logregressor)

#Second model
xgbclassifier(X_train,y_train,X_test,y_test)
#RoC curve for the model
ROC_curve(X_test,y_test,model)

#Third model
multilayerperceptron(X_train,y_train,X_test,y_test)
#RoC curve for the model
ROC_curve(X_test,y_test,mlp)

#Fourth model
RFClassifier(X_train,y_train,X_test,y_test)
#RoC curve for the model
ROC_curve(X_test,y_test,clf)