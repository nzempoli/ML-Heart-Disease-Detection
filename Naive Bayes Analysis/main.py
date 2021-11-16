import pandas
import numpy as np
#Name: Nicholas Zempolich
#Title: Milestone two, Binomial Naive Bayes
#Code is in Python and pandas must be installed
def increaseCategoryCount(data,property,classLabel,index,row): #Increases the count of a specific category attribute value for class classLabel
    property[classLabel][int(data.loc[row][index])] = property[classLabel][int(data.loc[row][index])] + 1

def calcMeanDeviation(data,property,classLabel,name): #calculates the mean and standard deviation for a single column/attribute in data
    property[classLabel][0] = data[name].mean()
    property[classLabel][1] = data[name].std()

def resetAttributes(attribute): #sets all values in an attribute item to zero for a new loop.
    for i in range(2):
        for j in range(len(attribute[i])):
            attribute[i][j]=0

def GaussianProbability(property,value,classLabel): #calculates Gaussian probability for a continous value
    return ((1/(np.sqrt(3.14*2)*property[classLabel][1]))*np.exp((-1/2) * ((value-property[classLabel][0])**2)/(2*(property[classLabel][1]**2))))

def categoryProbability(property,value,classLabel,totalClass): #Returns the probability of value given class classLabel
    return property[classLabel][value]/totalClass

def totalProbability(data, aC2, aC3, aC6, aC7, aC9, aC11, aC12, aC13, ag1, ag4, ag5, ag8, ag10,classLabel,totalClass,total,i): #calculates the total probability of classLabel for data
    prob1 = GaussianProbability(ag1,int(data.loc[i][0]),classLabel)  #Calculates probability density
    prob2 = categoryProbability(aC2,int(data.loc[i][1]),classLabel,totalClass) #Divides the count of the value for the attribute by the total class
    prob3 = categoryProbability(aC3,int(data.loc[i][2]),classLabel,totalClass)
    prob4 = GaussianProbability(ag4,int(data.loc[i][3]),classLabel)
    prob5 = GaussianProbability(ag5,int(data.loc[i][4]),classLabel)
    prob6 = categoryProbability(aC6,int(data.loc[i][5]),classLabel,totalClass)
    prob7 = categoryProbability(aC7,int(data.loc[i][6]),classLabel,totalClass)
    prob8 = GaussianProbability(ag8,int(data.loc[i][7]),classLabel)
    prob9 = categoryProbability(aC9,int(data.loc[i][8]),classLabel,totalClass)
    prob10 = GaussianProbability(ag10,int(data.loc[i][9]),classLabel)
    prob11 = categoryProbability(aC11,int(data.loc[i][10]),classLabel,totalClass)
    prob12 = categoryProbability(aC12, int(data.loc[i][11]), classLabel, totalClass)
    prob13 = categoryProbability(aC13, int(data.loc[i][12]), classLabel, totalClass)
    xByC = prob1*prob2*prob3*prob4*prob5*prob6*prob6*prob7*prob8*prob9*prob10*prob11*prob12*prob13 #calculates observed probability
    classChance = totalClass/total #calculates  prior probability for class classLabel
    return xByC*classChance

def trainProbabilities(data,aC2, aC3, aC6, aC7, aC9, aC11, aC12, aC13,ag1,ag4,ag5,ag8,ag10,classLabel): #Trains the Naives Bayes algorithm for class classLabel
    for i in data.index: #For each tuple in data
        increaseCategoryCount(data,aC2,classLabel,1, i) #Increase the appropriate category value count
        increaseCategoryCount(data, aC3, classLabel, 2, i)
        increaseCategoryCount(data, aC6, classLabel, 5, i)
        increaseCategoryCount(data, aC7, classLabel, 6, i)
        increaseCategoryCount(data, aC9, classLabel, 8, i)
        increaseCategoryCount(data, aC11, classLabel, 10, i)
        increaseCategoryCount(data, aC12, classLabel, 11, i)
        increaseCategoryCount(data, aC13, classLabel, 12, i)
    calcMeanDeviation(data,ag1,classLabel,"age") #or calculate the mean and standard deviation
    calcMeanDeviation(data, ag4, classLabel, "trestbps")
    calcMeanDeviation(data, ag5, classLabel, "chol")
    calcMeanDeviation(data, ag8, classLabel, "thalach")
    calcMeanDeviation(data, ag10, classLabel, "oldpeak")


def AutoCreateCategoryList(length): #Creates a 2d array, with each 1d array as length length
    list = [[0]*length,[0]*length]
    return list

def testProbabilities(data, aC2, aC3, aC6, aC7, aC9, aC11, aC12, aC13, ag1, ag4, ag5, ag8, ag10,totalClass0,totalClass1,total): #given a trained algorithm compare to test
    totalCorrect = 0
    totalWrong = 0
    classified = 0
    TP=0
    TN=0
    FP=0
    FN=0
    for i in data.index: #For each tuple in data
        pC0 = totalProbability(data, aC2, aC3, aC6, aC7, aC9, aC11, aC12, aC13, ag1, ag4, ag5, ag8, ag10,0,totalClass0,total,i) #calculate posterior of class 0
        pC1 = totalProbability(data, aC2, aC3, aC6, aC7, aC9, aC11, aC12, aC13, ag1, ag4, ag5, ag8, ag10,1,totalClass1, total, i)#calculate posterior of class 1
        if (pC0>pC1): #if posterior probability of class 0 is higher evaluate as class 0
            classified=0
        else: #otherwise evaluate as 1
            classified=1
        if classified == int(data.loc[i][13]): #if evalauted class is same as actual
            totalCorrect = totalCorrect + 1 #label as correct and update either TP or TN
            if classified == 1:
                TP = TP+1
            else:
                TN = TN+1
        else: #Otherwise add to the list of wrong values, and increase either FN or FP
            totalWrong = totalWrong + 1
            if classified==0:
                FN = FN+1
            else:
                FP = FP+1
    return [TP,TN,FN,FP]





def run_Bayes():  # Primary function to train Naive Bayes algorithms.
    #a - stands for attribute
    # The second character specifies the type with c being category and g being continuous
    ac2 = AutoCreateCategoryList(2)
    ac3 = AutoCreateCategoryList(4)
    ac6 = AutoCreateCategoryList(2)
    ac7 = AutoCreateCategoryList(3)
    ac9 = AutoCreateCategoryList(2)
    ac11 = AutoCreateCategoryList(3)
    ac12 = AutoCreateCategoryList(5)
    ac13 = AutoCreateCategoryList(4)
    ag1 = AutoCreateCategoryList(2)
    ag4 = AutoCreateCategoryList(2)
    ag5 = AutoCreateCategoryList(2)
    ag8 = AutoCreateCategoryList(2)
    ag10 = AutoCreateCategoryList(2)

    data = pandas.read_csv('KaggleDataSet.csv') #reads data from the dataset into a dataframe object
    accuracy = 0
    precision=0
    recall=0
    fMeasure=0
    randomseed=0             #randomseed variable, used to populate random_state
    loopcount=1             #Determines how many times the system loops over data
    resultsarray = [0,0,0,0] #initializes the array to return the results of classification
    seedarray = [96,34,60,61,179,166] #Seed array to utilize when replicating results in milestone 2

    for i in range(loopcount):
        randomseed = np.random.randint(250) #seedarray[i]
        train = data.sample(frac=0.8, random_state=randomseed)  # randomly splits the data based off randomSeed. Can be set manually to replicate results
        test = data.drop(train.index) #creates the test dataframe, which is the set of data in data NOT used in train
        trainClass1 = train[train['target'] == 1] #Divides train into two different classes based on the target/class column
        trainClass0 = train[train['target'] == 0]
        totalClass0 = len(trainClass0)
        totalClass1 = len(trainClass1)
        total = totalClass1+totalClass0 #Get the total number of values within the test class
        trainProbabilities(trainClass0, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10, 0) #Train the data off class 0
        trainProbabilities(trainClass1, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10, 1) #Train the data off class 0
        resultsarray = testProbabilities(test, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10,totalClass0,totalClass1,total) #fill results array with the TP,TP,FP,FN values
        resetAttributes(ac2) #when looping through multiple random seeds category values need to be reset to zero as they only count and can't reset on their own.
        resetAttributes(ac3)
        resetAttributes(ac6) #This does not have to be done for continous values as they're recalculated each loop.
        resetAttributes(ac7)
        resetAttributes(ac9)
        resetAttributes(ac11)
        resetAttributes(ac12)
        resetAttributes(ac13)
        TP=resultsarray[0]
        TN=resultsarray[1]
        FP=resultsarray[2]
        FN=resultsarray[3]
        print("Generated using random seed: {}".format(randomseed))
        accuracy = (TP+TN)/(TP+TN+FN+FP) *100
        precision = TP/(TP+FP)
        recall = (TP)/(TP+FN)
        fMeasure = (2 * precision * recall) / (precision +recall)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F-Measure: {}".format(fMeasure))

def run_setValidation():  # function that train NaiveBayes off tiny data sets.
    #a - stands for attribute
    # The second character specifies the type with c being category and g being continuous
    ac2 = AutoCreateCategoryList(2)
    ac3 = AutoCreateCategoryList(4)
    ac6 = AutoCreateCategoryList(2)
    ac7 = AutoCreateCategoryList(3)
    ac9 = AutoCreateCategoryList(2)
    ac11 = AutoCreateCategoryList(3)
    ac12 = AutoCreateCategoryList(5)
    ac13 = AutoCreateCategoryList(4)
    ag1 = AutoCreateCategoryList(2)
    ag4 = AutoCreateCategoryList(2)
    ag5 = AutoCreateCategoryList(2)
    ag8 = AutoCreateCategoryList(2)
    ag10 = AutoCreateCategoryList(2)

    data = pandas.read_csv('KaggleDataSet.csv') #reads data from the dataset into a dataframe object
    accuracy = 0
    precision=0
    randomseed=np.random.randint(250)
    number = 3 # four rounds of validation
    train = data.sample(frac=0.75, random_state=randomseed)
    test = data.drop(train.index)
    train1 = train.sample(frac=.33)
    trainholder = train.drop(train1.index)
    train2 = trainholder.sample(frac=.5)
    train3 = trainholder.drop(train2.index)


    for i in range(number):
        if i == 0:
            train=train1
        elif i == 1:
            train=train2
        else:
            train=train3
        trainClass1 = train[train['target'] == 1] #Divides train into two different classes based on the target/class column
        trainClass0 = train[train['target'] == 0]
        totalClass0 = len(trainClass0)
        totalClass1 = len(trainClass1)
        total = totalClass1+totalClass0 #Get the total number of values within the test class
        trainProbabilities(trainClass0, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10, 0) #Train the data off class 0
        trainProbabilities(trainClass1, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10, 1) #Train the data off class 0
        resultsarray = testProbabilities(test, ac2, ac3, ac6, ac7, ac9, ac11, ac12, ac13, ag1, ag4, ag5, ag8, ag10,totalClass0,totalClass1,total) #fill results array with the TP,TP,FP,FN values
        resetAttributes(ac2) #when looping through multiple random seeds category values need to be reset to zero as they only count and can't reset on their own.
        resetAttributes(ac3)
        resetAttributes(ac6) #This does not have to be done for continous values as they're recalculated each loop.
        resetAttributes(ac7)
        resetAttributes(ac9)
        resetAttributes(ac11)
        resetAttributes(ac12)
        resetAttributes(ac13)
        TP=resultsarray[0]
        TN=resultsarray[1]
        FP=resultsarray[2]
        FN=resultsarray[3]
        print("Generated using random seed: {}".format(randomseed))
        accuracy = (TP+TN)/(TP+TN+FN+FP) *100
        precision = TP/(TP+FP)
        recall = (TP)/(TP+FN)
        fMeasure = (2 * precision * recall) / (precision +recall)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F-Measure: {}".format(fMeasure))

run_Bayes()
#run_setValidation()

