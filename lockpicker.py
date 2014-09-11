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
import uuid
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectPercentile, f_classif
from time import gmtime, strftime

#define class variable for classifier
class _clf:
    def __init__(self, clf, indices, featuresTF, featureNames, clfName):
        self.clf = clf
        self.indicies = indices
        self.featuresTF = featuresTF
        self.featureNames = featureNames
        self.clfName = clfName
        


#read in train and test data

def readData(trainData, testData):
    if isinstance(trainData, str):
        train = pd.read_csv(trainData)
        test = pd.read_csv(testData)
        return train, test
    else:
        return trainData, testData

#create extra features
def createExtraFeatures():
    1
    #create extra features

    #split training into train and cv sets
def splitTrainingToCV(trainData, ycol, percentage):
     cols = [col for col in trainData.columns if col not in ycol]

     trainTrainData_X, trainCVData_X, trainTrainData_y,  trainCVData_y = train_test_split(\
                                                                        trainData[cols],\
                                                                        trainData[ycol], test_size=percentage)
     trainTrainData_X = pd.DataFrame(trainTrainData_X, columns = cols)
     trainCVData_X = pd.DataFrame(trainCVData_X, columns = cols)
     return trainTrainData_X, trainTrainData_y, trainCVData_X, trainCVData_y
#read total rows of all data and spit back row count
def numberOfFolds(percentage=0.02):
    return round(1.0/percentage)

#create kfolds for training
def createFolds(dfTrain, foldCount, stratified = 0):
    #nrows = len(dfTrain)
    if stratified == 0:
        kf = KFold(dfTrain, n_folds=foldCount, indices = True, random_state=2137)       
    else:
        kf = StratifiedKFold(dfTrain, n_folds=foldCount, indices = True)#, random_state=1)
    return kf

#create PCA
def createPCAFeatures():
    1
#check if labels are numeric, if not encode them
def encodeLabels():
    1
#create list of classifier with indices
def createClassifiers(clfType, indices, X_train, y_train, features=0, findTop=1, findTopPercentile=10):
    tempclf = []
    tclf = []
    clfName = uuid.uuid1()
    if clfType == "gbm":
        tclf = GradientBoostingClassifier(n_estimators=100)
    else:
        tclf = ExtraTreesClassifier(n_estimators=100, criterion='entropy', bootstrap=True)

    if findTop != 1:
        tempclf = _clf(tclf, indices, [x in features for x in X_train.columns], features, clfName)
    else:
        topFeatures = findTopFeatures(X_train, y_train, findTopPercentile)
        topFeatureNames = getFeatureNames(X_train.columns, topFeatures)
        tempclf = _clf(tclf, indices, topFeatures, topFeatureNames, clfName)

    tempclf.clf = trainClassifiersOnSelectedFeatures(X_train[tempclf.featureNames], y_train, tempclf.clf)
    return tempclf

def createClassifierGroup(clfType, X_train, y_train, thresholds, folds, features=0, findTop=1 ):
    clfgroup = []

    if findTop == 1:
        clfgroup = [createClassifiers(clfType, test_index, X_train.ix[test_index,:], y_train[test_index], features, findTop, p) for train_index, test_index in folds for p in thresholds]
    else:
        clfgroup = [createClassifiers(clfType, test_index, X_train.ix[test_index,:], y_train[test_index], features, findTop, p) for train_index, test_index in folds for p in thresholds]

    return clfgroup

def createClassifierPlatoon(X_train, y_train, thresholds, folds, selectedFeatures=0):
    clfs = []
    gbmsTop = createClassifierGroup("gbm", X_train, y_train, thresholds, folds, 0, 1)
    etcsTop = createClassifierGroup("etc", X_train, y_train, thresholds, folds, 0, 1)
    clfs.append(gbmsTop)
    clfs.append(etcsTop)

    if selectedFeatures != 0:
        gbmsSelect = createClassifierGroup("gbm", X_train, y_train, thresholds, folds, selectedFeatures, 0)
        etcsSelect = createClassifierGroup("etc", X_train, y_train, thresholds, folds, selectedFeatures, 0)

        clfs.append(gbmsSelect)
        clfs.append(etcsSelect)

    return np.ravel(clfs)

def featureCorrelationMatrix():
    1
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
    clf.fit(X,y)
    return clf

#pickle based on time and name
def saveClassifiers(clfs, name):
    dt = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    saveName = name + "--" + dt + ".pkl"
    joblib.dump(clfs, saveName)

def createEnsembleFrame(clfs, X):
    ensembleFrame = pd.DataFrame([createPrediction(c.clf,X.ix[:,c.featureNames]) for c in clfs])
    return ensembleFrame

#create prediction
def createEnsembler(method, X_train, y_train, X_test, y_test):
    
    if method not in ["gbm", "etc", "average"]:
        raise NameError("Please select either 'gbm', 'etc', or 'average'")
        

    if method == "average":
        yPred = X_test.mean(axis=1)
        aucmetric(y_test,yPred)
        return

    if method == "gbm":
        eclf =  GradientBoostingClassifier(n_estimators=1000)
        eclf.fit(X_train, y_train)
        yPred = eclf.predict_proba(X_test)
        aucmetric(y_test,yPred)
    else:
        eclf = ExtraTreesClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
        eclf.fit(X_train, y_train)
        yPred = eclf.predict_proba(X_test)
        aucmetric(y_test,yPred)
    return eclf

def createPrediction(clf, X, method=""):
    #print clf
    #print X.head()
    if method == "average":
        pred = X.mean(axis=1)
    else:
        pred = clf.predict_proba(X)[:,1]
    return pred
#main

def aucmetric(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(np.ravel(y), pred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print "AUC: ", auc

def main(trainData, testData, ycol, idcol,  saveName, foldPercentage=0.02, cvPercentage=0.25, selectedFeatures=0,ensembleMethod="average", stratifiedFolds=1, seed = 42):
    print "Let the data lockpicking begin!"
    np.random.seed(seed)
    print "Reading Data"
    train, test = readData(trainData, testData)
    trainTrainData_X, trainTrainData_y, trainCVData_X, trainCVData_y = splitTrainingToCV(trainData, \
                                                                        ycol, cvPercentage)
    
    thresholds = [5,10,25,50]
    folds = createFolds(trainTrainData_X, numberOfFolds(foldPercentage), stratifiedFolds)

    clfs = createClassifierPlatoon(trainTrainData_X, trainTrainData_y, thresholds, folds, selectedFeatures)
    #return clfs
    saveClassifiers(clfs, saveName)

    ensembleFrameTrain = createEnsembleFrame(clfs, trainTrainData_X)
    ensembleFrameTest = createEnsembleFrame(clfs, trainCVData_X)

    ensembleCLF = createEnsembler(ensembleMethod, ensembeFrameTrain, trainTrainData_y, ensembleFrameTest, trainCVData_y)

    testFrame = createEnsembleFrame(clfs, test)
    prediction = createPrediction(ensembleCLF, testFrame, ensembleMethod)
    predictionDF = pd.DataFrame(test[idcol])
    predictionDF[train[ycol].columns] = prediction
    predictionDF.to_csv(saveName + "-" + strftime("%Y-%m-%d_%H_%M_%S", gmtime()) + ".csv")

    print "Good Luck!"




if __name__ == "main":
    main()