# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 08:42:46 2019

@author: Hassan
"""
import math
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle

# from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# from sklearn.decomposition import PCA


def euclideanDistance(x1, x2):
    return (math.sqrt(sum((x1 - x2)**2)))
 
#This function returns the indexes of rows in trainingset for k Nearest neighbours of  test instance   
def getNeighbors(xTrainSet, xTestInstance, k):
    dists = np.zeros(len(xTrainSet))
    n = np.zeros(k)
    for j in range(len(xTrainSet)):
        d = euclideanDistance(xTrainSet[j], xTestInstance)
        dists[j] = d
    sortedIndexes = np.argsort(dists)
    for i in range(k):
        n[i] = sortedIndexes[i] 
    return n

#This function predicts the class of test instance according to majority rule voting by k nearest neighbours class label
def predict(neighbors, y_train):
    count= 0
    for z in range(len(neighbors)):    
        if(y_train[np.int(neighbors[z])] == 1):
            count += 1
        else:
            count -= 1   
    if(count > 0):
        return 1
    elif(count < 0):
        return 0

class Dataset:
    def __init__(self, datapath):
        self.datapath = datapath
        self.data = []
        self.labels = []

    def read_data(self):
        df = pd.read_csv(self.datapath, header=None)
        num_features = df.shape[1] - 1
        y = df.iloc[:, -1].values
        y[y==-1] = 0
        self.labels = y
        self.data = df.iloc[:, :-1].values
        
        print(self.labels.shape)
        print(self.data.shape)
        print(np.count_nonzero(self.labels==1),\
              np.count_nonzero(self.labels==0))
        
    def shuffle_data(self):
        self.data, self.labels = shuffle(self.data, self.labels, random_state=550)

    def normalize_data(self, min_range, max_range):
        self.data = minmax_scale(self.data, feature_range=(min_range, max_range))

    def prepare(self, n_splits=10, normalize=True, shuffle_data=True,\
                                       oversample=True, undersample=False):
        self.read_data()
        if oversample:
            ros = RandomOverSampler(random_state=55)
            self.data, self.labels = ros.fit_resample(self.data, self.labels)
        elif undersample:
            rus = RandomUnderSampler(random_state=55)
            self.data, self.labels = rus.fit_resample(self.data, self.labels)
        if shuffle_data:
            self.shuffle_data()
        if normalize:
            self.normalize_data(0, 1)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_data, random_state=43)
        return skf
# read & split data 
dataset = Dataset('C:/Users/Hassan/Desktop/EE485_data/GitHub_Project/DATA_Project/Dataset/dataset.txt')
skf = dataset.prepare(n_splits=10, normalize=False, shuffle_data=True,\
                               oversample=True, undersample=False)

# separate test set (%10) using one of the folds
X_test = np.array([])
y_test = np.array([])
test_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
for train, test in test_skf.split(dataset.data, dataset.labels):
    dataset.data, X_test = dataset.data[train], dataset.data[test]
    dataset.labels, y_test = dataset.labels[train], dataset.labels[test]
    break

start = time.time()
X_train = dataset.data
y_train = dataset.labels

k = 1
fold = 1
cumm_valid_pred = np.array([])
cumm_valid_y = np.array([])

for train_index, test_index in skf.split(dataset.data, dataset.labels):

    X_train, X_valid = dataset.data[train_index], dataset.data[test_index]
    y_train, y_valid = dataset.labels[train_index], dataset.labels[test_index]
    val_predictions = np.zeros(len(X_valid))
    
    print("\nFold {}:\ttrain_size:{} valid_size:{}".format(fold, len(X_train), len(X_valid)))
    a = np.count_nonzero(y_train)
    b = np.count_nonzero(y_valid)
    print("Class Ratios(+1/-1):\ttrain:{}/{} valid:{}/{}\n".format(a, len(y_train)-a, b, len(y_valid)-b))
    
    for i in range(len(X_valid)):
        neighbors = getNeighbors(X_train, X_valid[i], k) 
        val_predictions[i] = predict(neighbors, y_train)

    print("Validation Size:",len(X_valid))
    a = np.count_nonzero(y_valid)
    print("Class Ratios(+1/-1):\t{}/{}".format(a, len(y_valid)-a))

    p, r, f1, sup = precision_recall_fscore_support(y_valid, val_predictions)
    total_acc = accuracy_score(y_valid, val_predictions)
    
    cumm_valid_pred = np.hstack((cumm_valid_pred, val_predictions))
    cumm_valid_y = np.hstack((cumm_valid_y, y_valid))
    
    print("Total accuracy is {0:.4f}".format(total_acc))
    for i in range(2):
        print("Class {} Accuracy: {:.2f}".format(2*i - 1, r[i]))
    
    fold += 1

print('\n=================')
print('Overall Valid Report:')
print(classification_report(cumm_valid_y, cumm_valid_pred, target_names=['Class -1', 'Class 1']))
print('Overall Valid Accuracy:', accuracy_score(cumm_valid_y, cumm_valid_pred))
print('=================\n')
        

# test time
test_predictions = np.zeros(len(X_test))
for i in range(len(X_test)):
    neighbors = getNeighbors(X_train, X_test[i], k) 
    test_predictions[i] = predict(neighbors, y_train)

print("Test Size:",len(X_test))
a = np.count_nonzero(y_valid)
print("Class Ratios(+1/-1):\t{}/{}".format(a, len(y_test)-a))

p, r, f1, sup = precision_recall_fscore_support(y_test, test_predictions)
total_acc = accuracy_score(y_test, test_predictions)
print("Test accuracy is {0:.4f}".format(total_acc))
for i in range(2):
    print("Class {} Accuracy: {:.2f}".format(2*i - 1, r[i]))
