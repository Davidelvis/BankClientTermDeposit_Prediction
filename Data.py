#A function to load our data and split it into train and test datasets

#importing the libraries
from sklearn.model_selection import train_test_split
def split_data(data):
    X=data.drop(columns=['y','duration'])
    y=data['y']
    X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.1,random_state=1)

    return X_train,y_train,X_test,y_test