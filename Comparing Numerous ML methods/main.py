# Import required libraries
import pandas as pd #importing pandas and sklearn as objects
import sklearn as sk
from keras.models import Sequential#importing keras/tensorlfow models for Neural Networks
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression #importing Sklearn for Logistic Regression and SVMs
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

#This class will train a Neural Network via the keras Sequential Model
from tensorflow.python.keras.utils.np_utils import to_categorical


def trainTree(batch_size,iterations,hidden_layers):
    if (hidden_layers is None): #If hidden layers is empty treat it as a basic neural network
        hidden_layers=0
    df = pd.read_csv('dataSet.csv')
    target_column = ['target']
    predictors = list(set(list(df.columns)) - set(target_column)) #gets a list of only input values
    df[predictors] = df[predictors] / df[predictors].max() #normalizing values, if not done accuracy decreases
    X = df[predictors].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=120) #Splits data between training and test
    y_train = to_categorical(y_train) #Categorizes classifier to binary to match with model
    y_test = to_categorical(y_test)

    model = Sequential()
    if (hidden_layers==0): #Creates a one hidden layer mode
        model.add(Dense(6, activation='relu', input_dim=13))
        model.add(Dense(2, activation='softmax'))
    if (hidden_layers == 1): #creates a two hidden layer network
        model.add(Dense(8, activation='relu', input_dim=13))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(2, activation='softmax'))
    if (hidden_layers == 2): #creates a three hidden layer network
        model.add(Dense(6, activation='relu', input_dim=13))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #compiles model
    model.fit(X_train, y_train,batch_size=batch_size, epochs=iterations) #fits model
    results = model.evaluate(X_test, y_test, verbose=0) #evaluates the model against test values
    return results[1] #return accuracy

# Given a random state and a C value makes a logistRegression model
def logisticRegression(initializer,regularizationStrength):
    df = pd.read_csv('dataSet.csv')
    target_column = ['target']
    predictors = list(set(list(df.columns)) - set(target_column))
    df[predictors] = df[predictors] / df[predictors].max()  # normalizing values, if not done accuracy decrease
    X = df[predictors].values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=120)
    xt = X_train.reshape(-1,13); #Transposes X
    model = LogisticRegression(solver='liblinear',C=regularizationStrength, random_state=initializer,fit_intercept=False)
    model.fit(xt, y_train.ravel())
    xtest = X_test.reshape(-1, 13);
    accuracy = model.score(xtest,y_test.ravel()) #gets accuracy
    return accuracy

def SVM(nonLinear):
    df = pd.read_csv('dataSet.csv')
    if (nonLinear == 2) : #generates an SVM based off polynomials
        svmBuild = svm.SVC(kernel='poly',probability=True)
    elif(nonLinear==1): #generates an SVM based off linear kernels
        svmBuild = svm.SVC(kernel='linear',probability=True)
    else:
        svmBuild = svm.SVC() #otherwise default to rbf
    target_column = ['target']
    predictors = list(set(list(df.columns)) - set(target_column))
    df[predictors] = df[predictors] / df[predictors].max()  # normalizing values, if not done accuracy decrease
    X = df[predictors].values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=120)
    svmBuild.fit(X_train,y_train.ravel()) #fits model
    testPredictions = svmBuild.predict(X_test) #gets results against test data
    return sk.metrics.accuracy_score(y_test, testPredictions) # returns accuracy

#gets the average of three training runs
def trainTreeAndGetAverage(batch_size,iterations,layers):
    accuracy=0
    for i in range (0,3):
        accuracy = accuracy + trainTree(batch_size,iterations,layers)
    return accuracy/3

if __name__ == '__main__':
    iterationaccuracy = [0]*5
    iterationaccuracy[0] = trainTreeAndGetAverage(100,10,2)
    iterationaccuracy[1] = trainTreeAndGetAverage(100, 30, 2)
    iterationaccuracy[2] = trainTreeAndGetAverage(100, 50, 2)
    iterationaccuracy[3] = trainTreeAndGetAverage(100, 70, 2)
    iterationaccuracy[4] = trainTreeAndGetAverage(100, 90, 2)
    batch_sizeAccuracy = [0]*5
    batch_sizeAccuracy[0] = trainTreeAndGetAverage(50, 50, 2)
    batch_sizeAccuracy[1] = trainTreeAndGetAverage(100, 50, 2)
    batch_sizeAccuracy[2] = trainTreeAndGetAverage(150, 50, 2)
    batch_sizeAccuracy[3] = trainTreeAndGetAverage(200, 50, 2)
    batch_sizeAccuracy[4] = trainTreeAndGetAverage(250, 50, 2)
    layerAccuracy = [0]*3
    layerAccuracy[0] = trainTreeAndGetAverage(150,50,0)
    layerAccuracy[1] = trainTreeAndGetAverage(150, 50, 1)
    layerAccuracy[2] = trainTreeAndGetAverage(150, 50, 2)
    averageRegression = [0]*6
    averageRegression[0] = (logisticRegression(100, .001) + logisticRegression(100, .001) + logisticRegression(100, .001)) / 3
    #averageRegression[1] = (logisticRegression(100, .01) + logisticRegression(100, .01) + logisticRegression(100, .01)) / 3
    #averageRegression[2] = (logisticRegression(100,.1) + logisticRegression(100,.1) + logisticRegression(100,.1))/3
    #averageRegression[3] = (logisticRegression(100,1) + logisticRegression(100,1) + logisticRegression(100,1))/3
    #averageRegression[4] = (logisticRegression(100,10) + logisticRegression(100,10) + logisticRegression(100,10))/3
    #averageRegression[5] = (logisticRegression(100, 100) + logisticRegression(100, 100) + logisticRegression(100, 100)) / 3
    averageRBFKernel = (SVM(0)+SVM(0)+SVM(0))/3
    linearKernel = (SVM(1)+SVM(1)+SVM(1))/3
    nonLinearKernel = (SVM(2)+SVM(2)+SVM(2))/3
# See PyCharm help at https://www.jetbrains.com/help/pycharm/



