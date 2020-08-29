#importing Libraries that will be used
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from Data import split_data

def logisticregressor(X_train,y_train,X_test,y_test):
     # create an object of the Logistic Regression Classifier Model
    regressor=LogisticRegression()
    # fit model with training datasets
    model_logregressor = regressor.fit(X_train, y_train)
    # make predictions 
    y_pred = regressor.predict(X_test)
    # evaluate predictions
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(regressor.score(X_test, y_test)))


def xgbclassifier(X_train,y_train,X_test,y_test):
    # fit model with training datasets
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions 
    y_pred = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of XGBClassifier on the test set: %.2f%%" % (accuracy * 100.0))

def multilayerperceptron(X_train,y_train,X_test,y_test):
    # create an object of the Multilayer Perceptron Model
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    
    # fit the model with the training data
    mlp.fit(X_train,y_train)
    # make predictions 
    predict_test = mlp.predict(X_test)

def RFClassifier(X_train,y_train,X_test,y_test):
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets 
    clf.fit(X_train,y_train)
    # make predictions 
    RF_pred=clf.predict(X_test)

def ROC_curve(X_test,y_test,model_type):
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = model_type.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
