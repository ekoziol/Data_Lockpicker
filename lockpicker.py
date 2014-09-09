'''
Created by Eric Koziol
@erickoziol
September 7, 2014
V0.1
Lockpicker is designed to be a brute force ensembler
The data is heavily undersampled by taking the test indices of each fold.
The default fold size is 2 percent of the number of rows.
'''

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
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
def splitTrainingToCV(trainData, ycol, percentage):
     trainTrainData_X, trainTrainData_y, trainCVData_X, trainCVData_y = train_test_split(\
                                                                        trainData[trainData.columns not in ycol],\
                                                                        trainData[ycol], random_state=1)
     return trainTrainData_X, trainTrainData_y, trainCVData_X, trainCVData_y
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
        tempclf = _clf(tclf, indices, topFeatures, topFeatureNames)

    tempclf.clf = trainClassifiersOnSelectedFeatures(X_train[tempclf.FeatureNames], y_train, tempclf.clf)
    return tempclf

def createClassifierGroup(clfType, indices, X_train, y_train, thresholds, folds, features=0, findTop=1 ):
    clfgroup = []

    if findTop == 1:
        clfgroup = [createClassifiers(clfType, indices, X_train[test_index], y_train[test_index], \
                        features=0, findTop=1, findTopPercentile=p) for train_index, test_index in folds for p in thresholds]
    else:
        clfgroup = [createClassifiers(clfType, indices, X_train[test_index], y_train[test_index], \
                        features, findTop=0, findTopPercentile=p) for train_index, test_index in folds forr p in thresholds]

    return clfgroup

def createClassifierPlatoon(X_train, y_train, thresholds, folds, indices, selectedFeatures=0):
    clfs = []
    gbmsTop = createClassifierGroup("gbm", indices, X_train, y_train, thresholds, folds, 0, 1)
    etcsTop = createClassifierGroup("etc", indices, X_train, y_train, thresholds, folds, 0, 1)
    clfs.append(gbmsTop)
    clfs.append(etcsTop)

    if selectFeatures != 0:
        gbmsSelect = createClassifierGroup("gbm", indices, X_train, y_train, thresholds, folds, selectFeatures, 0)
        etcsSelect = createClassifierGroup("etc", indices, X_train, y_train, thresholds, folds, selectFeatures, 0)

        clfs.append(gbmsSelect)
        clfs.append(etcsSelect)

    return clfs

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
def trainClassifiersOnSelectedFeatures(X, y, clf):
    clf.clf.fit(X,y)
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



    if method == "gbm":
        eclf = tclf = GradientBoostingClassifier(n_estimators=1000)
    else if method == "etc":
        eclf = ExtraTreesClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
    else:

def createEnsemblePrediction(clf, X):
    pred = clf.predict_proba(X)
    return pred
#main


def main(trainData, testData, ycol, foldPercentage, cvPercentage=0.25, selectedFeatures=0,ensembleMethod="average", stratifiedFolds=1):
    print "Let the data lockpicking begin!"
    np.seed(42)
    print "Reading Data"
    train, test = readData(trainData, testData)
    trainTrainData_X, trainTrainData_y, trainCVData_X, trainCVData_y = splitTrainingToCV(trainData, \
                                                                        ycol, cvPercentage)
    
    thresholds = [5,10,25,50]
    folds = createFolds(trainData, numberOfFolds(foldPercentage), stratifiedFolds)

    clfs = createClassifierPlatoon(X_train, y_train, thresholds, folds, indices, selectedFeatures)
    


if __name__ == "main":
    main()