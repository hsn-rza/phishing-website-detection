# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 08:14:45 2019

@author: Hassan
"""

import pandas as pd
#import matplotlib.pyplot as plot
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
# from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def calc_accuracy(y, predictions):
    y = y.squeeze()
    preds = predictions.squeeze()
    p, r, f1, sup = precision_recall_fscore_support(y, preds)
    total_acc = accuracy_score(y, preds)
    class_based_accuracies = r  
    return total_acc, class_based_accuracies

def print_accuracy(total_acc, class_based_accuracies):
    num_of_classes = class_based_accuracies.shape[0]
    print("Total accuracy is {0:.4f}".format(total_acc))
    for i in range(num_of_classes):
        print("Class {} Accuracy: {:.4f}".format(2*i - 1, class_based_accuracies[i]))
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

def logistic_func(X, weights):
    z = np.dot(X,weights)
    return 1 / (1 + np.exp(-z))

def g_ascent(X, h, y, weight, rate):
    return weight + rate * (1/len(X))*np.dot(X.T, y - h)

def full_batch(x_train, y_train, weights, rate, max_iters, cumm_train_y, cumm_train_pred):
    for i in range(max_iters):
#        print('FB Iter:', i+1)
        h = logistic_func(x_train, weights)
        weights = g_ascent(x_train, h, y_train, weights, rate)
        
        # display train accuracy
        _, preds = predict(x_train,y_train, weights)
#        total_acc, class_based_accuracies = calc_accuracy(y_train, preds)
#        print_accuracy(total_acc, class_based_accuracies)
#        print()
    cumm_train_y['FB'] = np.hstack( (cumm_train_y['FB'], y_train.squeeze()))
    cumm_train_pred['FB'] = np.hstack( (cumm_train_pred['FB'], preds.squeeze()))
    return weights

def mini_batch(x_train, y_train, weights, rate, batch_size, max_iters, cumm_train_y, cumm_train_pred):
    for k in range(max_iters):
#        print('MB Iter:', k+1)
        for i in range(int(len(x_train)/(batch_size))):
            x_train_MB = x_train[i*(batch_size):(i+1)*(batch_size), :]
            y_train_MB = y_train[i*(batch_size):(i+1)*(batch_size), :]
            h = logistic_func(x_train_MB, weights)
            weights = weights + rate * (1/(batch_size)*np.dot(x_train_MB.T, y_train_MB - h))
        
        if len(x_train) % batch_size != 0:
            i = int(len(x_train)/(batch_size))
            x_train_MB = x_train[i*(batch_size):, :]
            y_train_MB = y_train[i*(batch_size):, :]
            h = logistic_func(x_train_MB, weights)
            weights = weights + rate * (1/(x_train_MB.shape[0])*np.dot(x_train_MB.T, y_train_MB - h))
            
         # display train accuracy
        _, preds = predict(x_train,y_train, weights)
#        total_acc, class_based_accuracies = calc_accuracy(y_train, preds)
#        print_accuracy(total_acc, class_based_accuracies)
#        print()
    cumm_train_y['MB'] = np.hstack( (cumm_train_y['MB'], y_train.squeeze()))
    cumm_train_pred['MB'] = np.hstack( (cumm_train_pred['MB'], preds.squeeze()))
    return weights

def stochastic(x_train, y_train, weights, rate, max_iters, cumm_train_y, cumm_train_pred):
    for k in range(max_iters):
#        print('ST Iter:', k+1)
        for i in range(len(x_train)):
            x_train_ST = x_train[i,:].reshape(len(x_train[0]),1)
            y_train_ST = y_train[i,:].reshape(1,1)
            z = x_train_ST.T@weights
            h = 1 / (1 + np.exp(-z))
            weights = weights + rate*x_train_ST*(y_train_ST-h)

         # display train accuracy
        _, preds = predict(x_train,y_train, weights)
#        total_acc, class_based_accuracies = calc_accuracy(y_train, preds)
#        print_accuracy(total_acc, class_based_accuracies)
#        print()
    cumm_train_y['ST'] = np.hstack( (cumm_train_y['ST'], y_train.squeeze()))
    cumm_train_pred['ST'] = np.hstack( (cumm_train_pred['ST'], preds.squeeze()))
    return weights
    
def predict(x_test,y_test, trained_weights):
    y_predicted = logistic_func(x_test, trained_weights)
    y_predicted[y_predicted < 0.5] = 0
    y_predicted[y_predicted >= 0.5] = 1
    true_predictions_count = sum(1*(y_predicted == y_test))
    accuracy = (true_predictions_count/x_test.shape[0])*100
    return accuracy,y_predicted

X_test = np.array([])
y_test = np.array([])

def train(k, lr, batch_size, max_iters, onlyMB=False):
    global X_test, y_test
    
    cumm_train_pred = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}
    cumm_train_y = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}
    cumm_valid_pred = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}
    cumm_valid_y = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}

    # read & split data 
    dataset = Dataset('C:/Users/Hassan/Desktop/EE485_data/DATA_Project/dataset.txt')
    skf = dataset.prepare(n_splits=k, normalize=False, shuffle_data=True,\
                                   oversample=True, undersample=False)
                                   
    # separate test set (%10) using one of the folds
    X_test = np.array([])
    y_test = np.array([])
    test_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
    for train, test in test_skf.split(dataset.data, dataset.labels):
        dataset.data, X_test = dataset.data[train], dataset.data[test]
        dataset.labels, y_test = dataset.labels[train], dataset.labels[test]
        break
    y_test = y_test.reshape(-1,1)

    tot_FB_time = 0.
    tot_MB_time = 0.
    tot_ST_time = 0.

    i = 1
    for train_index, test_index in skf.split(dataset.data, dataset.labels):
        X_train, X_valid = dataset.data[train_index], dataset.data[test_index]
        y_train, y_valid = dataset.labels[train_index], dataset.labels[test_index]
        
        y_train, y_valid = y_train.reshape(-1,1), y_valid.reshape(-1,1)
        
        print("\nFold {}:\ttrain_size:{} valid_size:{}".format(i, len(X_train), len(X_valid)))
        a = np.count_nonzero(y_train)
        b = np.count_nonzero(y_valid)
        print("Class Ratios(+1/-1):\ttrain:{}/{} valid:{}/{}\n".format(a, len(y_train)-a, b, len(y_valid)-b))
        
        FB_train_weights = -1
        if not onlyMB:
            # Full Batch
            weights = np.zeros(len(X_valid[0])).reshape(len(X_train[0]),1)
            start = time.time()
            FB_train_weights = full_batch(X_train, y_train, weights, lr, max_iters, cumm_train_y, cumm_train_pred)
            end = time.time()
            total_time = end - start
            tot_FB_time += total_time
            
            FB_accuracy = predict(X_valid,y_valid, FB_train_weights)
            predictions = FB_accuracy[1]
            total_acc, class_based_accuracies = calc_accuracy(y_valid, predictions)
            print_accuracy(total_acc, class_based_accuracies)
            print("The accuracy for Full Batch GD is %f with training time of %f seconds." % (FB_accuracy[0],total_time))
            cumm_valid_pred['FB'] = np.hstack( (cumm_valid_pred['FB'], predictions.squeeze()))
            cumm_valid_y['FB'] = np.hstack( (cumm_valid_y['FB'], y_valid.squeeze()))

        # Mini Batch
        weights = np.zeros(len(X_valid[0])).reshape(len(X_train[0]),1)
        start = time.time()
        MB_train_weights = mini_batch(X_train, y_train, weights, lr, batch_size, max_iters, cumm_train_y, cumm_train_pred)
        end = time.time()
        total_time = end - start
        tot_MB_time += total_time
        
        MB_accuracy = predict(X_valid,y_valid, MB_train_weights)
        predictions = MB_accuracy[1]
        total_acc, class_based_accuracies = calc_accuracy(y_valid, predictions)
        print_accuracy(total_acc, class_based_accuracies)
        print("The accuracy for Mini Batch GD is %f with training time of %f seconds." % (MB_accuracy[0],total_time))
        cumm_valid_pred['MB'] = np.hstack( (cumm_valid_pred['MB'], predictions.squeeze()))
        cumm_valid_y['MB'] = np.hstack( (cumm_valid_y['MB'], y_valid.squeeze()))
        
        ST_train_weights = -1
        if not onlyMB:
            # Stochastic
            weights = np.zeros(len(X_valid[0])).reshape(len(X_train[0]),1)
            start = time.time()
            ST_train_weights = stochastic(X_train, y_train, weights, lr, max_iters, cumm_train_y, cumm_train_pred)
            end = time.time()
            total_time = end - start
            tot_ST_time += total_time
            
            ST_accuracy = predict(X_valid,y_valid, ST_train_weights)
            predictions = ST_accuracy[1]
            total_acc, class_based_accuracies = calc_accuracy(y_valid, predictions)
            print_accuracy(total_acc, class_based_accuracies)
            print("The accuracy for Stochastic GD is %f with training time of %f seconds." % (ST_accuracy[0],total_time))
            cumm_valid_pred['ST'] = np.hstack( (cumm_valid_pred['ST'], predictions.squeeze()))
            cumm_valid_y['ST'] = np.hstack( (cumm_valid_y['ST'], y_valid.squeeze()))
        i += 1
        
    cumm_dicts = (cumm_train_pred, cumm_train_y, cumm_valid_pred, cumm_valid_y)
    trained_weights = (FB_train_weights, MB_train_weights, ST_train_weights)
    times = (tot_FB_time, tot_MB_time, tot_ST_time)

    return cumm_dicts, trained_weights, times

# test time
def print_test_results(GD, trained_weights, times):
    FB_train_weights = trained_weights[0]
    MB_train_weights = trained_weights[1]
    ST_train_weights = trained_weights[2]
    tot_FB_time = times[0]
    tot_MB_time = times[1]
    tot_ST_time = times[2]
    
    if GD == 'FB':
        FB_accuracy = predict(X_test,y_test, FB_train_weights)
        predictions = FB_accuracy[1]
        total_acc, class_based_accuracies = calc_accuracy(y_test, predictions)
        print_accuracy(total_acc, class_based_accuracies)
        print("Test accuracy for Full Batch GD is %f with total training time of %f seconds." % (FB_accuracy[0],tot_FB_time))
    elif GD == 'MB':
        MB_accuracy = predict(X_test,y_test, MB_train_weights)
        predictions = MB_accuracy[1]
        total_acc, class_based_accuracies = calc_accuracy(y_test, predictions)
        print_accuracy(total_acc, class_based_accuracies)
        print("Test accuracy for Mini Batch GD is %f with total training time of %f seconds." % (MB_accuracy[0],tot_MB_time))
    elif GD == 'ST':
        ST_accuracy = predict(X_test,y_test, ST_train_weights)
        predictions = ST_accuracy[1]
        total_acc, class_based_accuracies = calc_accuracy(y_test, predictions)
        print_accuracy(total_acc, class_based_accuracies)
        print("Test accuracy for Stochastic GD is %f with total training time of %f seconds." % (ST_accuracy[0],tot_ST_time))
    else:
        print("Unrecognized GD option ", GD)
        
    return total_acc

batch_sizes = [32, 64, 128]
learning_rates = [0.01, 0.001]

val_accuracies = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}
val_models = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}
test_accuracies = {'FB': np.array([]), 'MB': np.array([]), 'ST': np.array([])}

k = 10
max_iters = 100
for bs in batch_sizes:
    for lr in learning_rates:
        print('\n\n===========================')
        print('Batch Size:{}, Learning Rate:{}'.format(bs, lr))
        print('===========================\n')
        
        onlyMB=(bs!=batch_sizes[0])
        if not onlyMB:
            cumm_dicts, trained_weights, times = train(k=k, lr=lr, batch_size=bs, max_iters=max_iters, onlyMB=onlyMB)

            cumm_train_pred = cumm_dicts[0]
            cumm_train_y = cumm_dicts[1]
            cumm_valid_pred = cumm_dicts[2]
            cumm_valid_y = cumm_dicts[3]
            
            for GD in ['FB', 'MB', 'ST']:
                print('\n=================')
                print('Overall Train[{}] Report:'.format(GD))
                print(classification_report(cumm_train_y[GD], cumm_train_pred[GD], target_names=['Class -1', 'Class 1']))
                print('Overall Train[{}] Accuracy:'.format(GD), accuracy_score(cumm_train_y[GD], cumm_train_pred[GD]))
                print('=================')
                print('Overall Valid[{}] Report:'.format(GD))
                print(classification_report(cumm_valid_y[GD], cumm_valid_pred[GD], target_names=['Class -1', 'Class 1']))
                print('Overall Valid[{}] Accuracy:'.format(GD), accuracy_score(cumm_valid_y[GD], cumm_valid_pred[GD]))
                print('=================\n')

                val_accuracies[GD] = np.hstack( (val_accuracies[GD], accuracy_score(cumm_valid_y[GD], cumm_valid_pred[GD])))
                val_models[GD] = np.hstack( (val_models[GD], np.array({'batch_size': bs, 'lr': lr})))
                test_accuracy = print_test_results(GD, trained_weights, times)
                test_accuracies[GD] = np.hstack( (test_accuracies[GD], test_accuracy))
            
                
        else:   # only MB
            cumm_dicts, trained_weights, times = train(k=k, lr=lr, batch_size=bs, max_iters=max_iters, onlyMB=onlyMB)

            cumm_train_pred = cumm_dicts[0]
            cumm_train_y = cumm_dicts[1]
            cumm_valid_pred = cumm_dicts[2]
            cumm_valid_y = cumm_dicts[3]
            
            print('\n=================')
            print('Overall Train[{}] Report:'.format('MB'))
            print(classification_report(cumm_train_y['MB'], cumm_train_pred['MB'], target_names=['Class -1', 'Class 1']))
            print('Overall Train[{}] Accuracy:'.format('MB'), accuracy_score(cumm_train_y['MB'], cumm_train_pred['MB']))
            print('=================')
            print('Overall Valid[{}] Report:'.format('MB'))
            print(classification_report(cumm_valid_y['MB'], cumm_valid_pred['MB'], target_names=['Class -1', 'Class 1']))
            print('Overall Valid[{}] Accuracy:'.format('MB'), accuracy_score(cumm_valid_y['MB'], cumm_valid_pred['MB']))
            print('=================\n')
            
            val_accuracies['MB'] = np.hstack( (val_accuracies['MB'], accuracy_score(cumm_valid_y['MB'], cumm_valid_pred['MB'])))
            val_models['MB'] = np.hstack( (val_models['MB'], np.array({'batch_size': bs, 'lr': lr})))
            MB_test_accuracy = print_test_results('MB', trained_weights, times)
            test_accuracies['MB'] = np.hstack( (test_accuracies['MB'], MB_test_accuracy))

print('\n\n---------RESULTS---------')
for GD in ['FB', 'MB', 'ST']:
    t = zip(val_accuracies[GD], val_models[GD], test_accuracies[GD])
    sorted_models = sorted(t, key=lambda tup: tup[0], reverse=True)
    for m in sorted_models:
        print('[{}] Best validation model accuracy:'.format(GD), m[0])
        print('[{}] Batch Size:{}, Learning Rate:{}'.format(GD, m[1]['batch_size'], m[1]['lr']))
        print('[{}] Test accuracy of the best model:'.format(GD), m[2])
        break
    print('---')

print('\nOther models:')
for GD in ['FB', 'MB', 'ST']:
    t = zip(val_accuracies[GD], val_models[GD], test_accuracies[GD])
    sorted_models = sorted(t, key=lambda tup: tup[0], reverse=True)
    i = 0
    for m in sorted_models:
        if i == 0:
            i = 1
            continue
        print('[{}] validation model accuracy:'.format(GD), m[0])
        print('[{}] Batch Size:{}, Learning Rate:{}'.format(GD, m[1]['batch_size'], m[1]['lr']))
        print('[{}] Test accuracy of the model:'.format(GD), m[2])
        print()
    print('---')
