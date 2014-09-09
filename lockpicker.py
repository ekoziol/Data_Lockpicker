# I didn't have much time to work on this competition, 
# so I kind of brute forced my way through it, without deep exploratory studies 
# of the features.

# My submission uses 1000 individual estimators in total. 
# Half are GBMs and half are Extra Trees. I heavily undersampled the data. 
# Each estimator deals with approx 10K data points.

# As for the features, half of the estimators deal with only the contract related data. 
# The other half uses sklearn's feature selection to pick top 150 features 
# from contract + crime + geodem + weather. Note that each estimator runs its 
# own feature selection based on its 10K data points.

# There's one little trick I used, which I guess others have also done. 
# Instead of predicting the losses directly, I took the logarithm, and predicted on that.

# The total training + predicting time on my several years old laptop is around 5-6 hours.


import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectPercentile, f_classif
from time import gmtime, strftime

#define class variable for classifier
class _clf:
    def __init__(self, clf, indices, featuresTF, featureNames):
        self.clf = clf
        self.indicies = indices
        self.featuresTF = featuresTF
        self.featuresNames = featuresNames


#read in train and test data

def readData(trainData, testData):
    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    return train, test

#create extra features
def createExtraFeatures():
    #create extra features

#split training into train and cv sets
def splitTrainingToCV(trainData, percentage = 0.25):

#read total rows of all data and spit back row count
def numberOfFolds(percentage=0.02):
    return round(1.0/percentage)

#create kfolds for training
def createFolds(dfTrain, foldCount, stratified = 0):
    if stratified == 0:
        kf = KFold(len(dfTrain.index), n_folds=foldCount, indices = True, random_state=2137)       
    else:
        kf = StratifiedKFold(len(dfTrain.index), n_folds=foldCount, indices = True, random_state=2137)
    return kf

#create PCA
def createPCAFeatures():

#create list of classifier with indices
def createClassifiers(clfType, indices, X_train, y_train, features=0, findTop=1, findTopPercentile=10):
    tempclf = []
    tclf = []
    if clfType = "gbm":
        tclf = GradientBoostingClassifier(n_estimators=100)
    else:
        tclf = ExtraTreesClassifier(n_estimators=100, criterion='entropy', bootstrap=True)

    if findTop != 1:
        tempclf = _clf(tclf, indices, [x in features for x in X_train.columns], features)
    else:
        topFeatures = findTopFeatures(X_train, y_train, findTopPercentile)
        topFeatureNames = getFeatureNames(X_train.columns, topFeatures)
        tempclf = +clf(tclf, indices, topFeatures, topFeatureNames)

    return tempclf



def featureCorrelationMatrix():

#find top features
#train classifiers on top 5%, 10%, 25% and 50% of features
#return classifiers and list of features
def findTopFeatures(X, y, threshold):
    selector = SelectPercentile(f_classif, percentile=threshold)
    selector.fit(X, y)
    return selector.get_support()

def getFeatureNames(columns, selectVector):
    return columns[selectVector]

#train classifier based on selected features
def trainClassifieresOnSelectedFeatures(X, y, clf):
    clf.fit(X,y)
    return clf

#pickle based on time and name
def saveClassifiers(clfs, name):
    dt = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    saveName = name + "--" + dt + ".pkl"
    joblib.dump(clfs, saveName)

#create prediction
def createEnsembler(method, X_train, X_test, y_test):
    if method not in ["gbm", "etc", "average"]:
        print "Please select either 'gbm', 'etc', or 'average'"
        break

def createEnsemblePrediction():
    
#main


def main(trainData, testData, foldPercentage, cvThreshold, ensembleMethod="average", stratifiedFolds=1):
    print "Let the data lockpicking begin!"
    np.seed(42)
    print "Reading Data"
    train, test = readData(trainData, testData)

    
    thresholds = [5,10,25,50]
    folds = createFolds(trainData, numberOfFolds(foldPercentage), stratifiedFolds)

    clfs = []

if __name__ == "main":
    main()