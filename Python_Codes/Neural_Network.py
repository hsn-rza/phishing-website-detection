#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# # Dataset class

# In[2]:


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
        print(np.count_nonzero(self.labels==1),              np.count_nonzero(self.labels==0))
        
    def shuffle_data(self):
        self.data, self.labels = shuffle(self.data, self.labels, random_state=550)

    def normalize_data(self, min_range, max_range):
        self.data = minmax_scale(self.data, feature_range=(min_range, max_range))

    def prepare_nn(self, n_splits=10, normalize=True, shuffle_data=True, oversample=True, undersample=False):
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


# # ANN

# In[3]:


class ANN:
    def __init__(self, input_size, output_size, hidden_size, hidden_size2=None, batch_size=None, lr=0.001, activation='sigmoid'):
        # parameters
        np.random.seed(42)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation

        # weight
        self.w1 = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.hidden_size))
        self.w2 = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_size, self.output_size))  # weight output
        if hidden_size2:
            self.w2 = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_size, self.hidden_size2))
            self.w3 = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_size2, self.output_size))  # weight output

        # bias
        self.b1 = np.random.rand() # from a uniform distribution over [0, 1)
        self.b2 = np.random.rand() 
        self.b3 = np.random.rand()

        #history
        self.accuracy = []
        self.loss = []
        self.w1s = []
        self.w2s = []
        self.b1s = []
        self.b2s = []
        self.w3s = []
        self.b3s = []
        
    def to_onehot(self, y):
        targets = np.array(y).reshape(-1)
        n_classes = np.unique(y).size
        return np.eye(n_classes)[targets]

    def softmax(self, x):
        exps = np.exp(x - x.max())  # more stable softmax
        return exps / np.sum(exps, axis=1, keepdims=True)

    def softmax_derivative(self, predictions, y_onehot):
        return predictions - y_onehot

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def get_derivative(self, z):
        if self.activation == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation == 'tanh':
            return 1 - np.power(self.tanh(z), 2)
        elif self.activation == 'relu':
            return self.relu_derivative(z)
        else: # no softmax in the hidden layers
            print(self.activation, "is not supported in the hidden layer!")
            sys.exit(-1)

    def get_activation(self, z):
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'relu':
            return self.relu(z)
        else: # no softmax in the hidden layers
            print(self.activation, "is not supported in the hidden layer!")
            sys.exit(-1)

    def cross_entropy(self, Y, Y_hat):
        m = Y.shape[0]
        x = Y_hat[range(m), Y]
        correct_logprobs = -np.log(x)
        data_loss = np.sum(correct_logprobs)
        return 1./m * data_loss
    
    def calc_accuracy(self, y, predictions):
        preds = np.argmax(predictions, axis=1)
        p, r, f1, sup = precision_recall_fscore_support(y, preds)
        total_acc = accuracy_score(y, preds)
        class_based_accuracies = r
        
        return total_acc, class_based_accuracies

    def print_accuracy(self, total_acc, class_based_accuracies):
        num_of_classes = class_based_accuracies.shape[0]
        print("Total accuracy is {0:.2f}".format(total_acc))
        for i in range(num_of_classes):
            print("Class {} Accuracy: {:.2f}".format(2*i - 1, class_based_accuracies[i]))

    def dropout_forward(self, A, p):
        """
            randomly drop out/shut down neurons of activation layer
            with probability of p
        :param A: Activation Layer
        :param p: dropout keep probability
        :return: Activation layer after dropping neurons and dropout mask
        """
        d = np.random.rand(A.shape[0], A.shape[1])
        d = d < p
        A = np.multiply(A, d)  # shut down some neurons of activation layer
#         A /= p
        return A, d

    def dropout_backward(self, dA, p, d):
        """
            shut down the same neurons as during the forward propagation
        :param dA: derivative of activation layer
        :param p: dropout factor
        :param d: dropout mask
        :return:
        """
        dA = np.multiply(d, dA)
        dA /= p
        return dA

    def forward_propagation(self, data, dropout_keep=1, is_test=False):
        z1 = data.dot(self.w1) + self.b1
        a1 = self.get_activation(z1)
        if not is_test:
            a1, d = self.dropout_forward(a1, dropout_keep)
        z2 = a1.dot(self.w2) + self.b2
        if self.hidden_size2:
            a2 = self.get_activation(z2)
            last_layer = a2.dot(self.w3) + self.b3
        else:
            last_layer = z2
        predictions = self.softmax(last_layer)
        if is_test:
            return predictions
        else:
            return predictions, z1, z2, d

    def backward_propagation(self, data, label_matrix, predictions, z1, z2, d_mask, dropout_keep=1):
#         predictions, z1, z2, d_mask = self.forward_propagation(data, dropout_keep)
        if self.hidden_size2:
            a2 = self.get_activation(z2)
            dZ3 = self.softmax_derivative(predictions, label_matrix)
            dW3 = a2.T.dot(dZ3)
            dB3 = np.sum(dZ3, axis=0, keepdims=True)
            dA2 = dZ3.dot(self.w3.T)
            activation_der = self.get_derivative(z2)
            dZ2 = dA2 * activation_der
            self.w3 -= self.lr * dW3  # update weights
            self.b3 -= self.lr * dB3  # update bias
        else:
            dZ2 = self.softmax_derivative(predictions, label_matrix)
        a1 = self.get_activation(z1)
        dW2 = a1.T.dot(dZ2)
        dB2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.w2.T)
        dA1 = self.dropout_backward(dA1, dropout_keep, d_mask)
        activation_der = self.get_derivative(z1)
        dZ1 = dA1 * activation_der  # sigmoid derivative
        dW1 = np.dot(data.T, dZ1)
        dB1 = np.sum(dZ1, axis=0)

        # update weights and bias
        self.w2 -= self.lr * dW2
        self.b2 -= self.lr * dB2
        self.w1 -= self.lr * dW1
        self.b1 -= self.lr * dB1

#         return predictions

    def train(self, X, y, epochs, dropout_keep=1):
        y_onehot = self.to_onehot(y)
        for epoch in range(epochs):
            if self.batch_size:
                for i in range(int(len(X) / self.batch_size)):
                    lo = i * self.batch_size
                    hi = (i + 1) * self.batch_size
                    if (hi > len(X)):
                        break
                    batch_data = X[lo:hi]
                    batch_label_matrix = y_onehot[lo:hi]
                    predictions, z1, z2, d_mask = self.forward_propagation(batch_data, dropout_keep)
                    self.backward_propagation(batch_data, batch_label_matrix, predictions,                                              z1, z2, d_mask, dropout_keep)
            else:
                predictions, z1, z2, d_mask = self.forward_propagation(X, dropout_keep)
                self.backward_propagation(X, y_onehot, predictions,                                              z1, z2, d_mask, dropout_keep)
            
            # one more forward with the updated weights
            predictions, _, _, _ = self.forward_propagation(X, dropout_keep)
            loss = self.cross_entropy(y, predictions)
            total_acc, class_based_accuracies = self.calc_accuracy(y, predictions)

            self.loss.append(loss)
            self.accuracy.append(total_acc)
            self.w1s.append(self.w1)
            self.w2s.append(self.w2)
            self.b1s.append(self.b1)
            self.b2s.append(self.b2)
            if self.hidden_size2:
                self.w3s.append(self.w3)
                self.b3s.append(self.b3)

            if epoch % 100 == 0:
                print("Epoch #{} with loss {}".format(epoch, loss))
                self.print_accuracy(total_acc, class_based_accuracies)
        print("Epoch #{} with loss {}".format(epoch, loss))
        self.print_accuracy(total_acc, class_based_accuracies)
        
        update_stats('train', y, predictions)
        plt.plot(self.accuracy)
        plt.ylabel('accuracy')
        plt.show()

    def test(self, X, y, dropout_keep=1, isTest=False):
        index_min = np.argmax(np.array(self.accuracy))  # pick the best model with highest accuracy
        if self.hidden_size2:
            self.w3 = self.w3s[index_min] * dropout_keep
            self.b3 = self.b3s[index_min]
        self.w1 = self.w1s[index_min] * dropout_keep
        self.b1 = self.b1s[index_min]
        self.w2 = self.w2s[index_min] * dropout_keep
        self.b2 = self.b2s[index_min]

        predictions = self.forward_propagation(X, is_test=True)

        loss = self.cross_entropy(y, predictions)
        print("Testing with the highest accuracy model:\nLoss of {}".format(loss))
        total_acc, class_based_accuracies = self.calc_accuracy(y, predictions)
        self.print_accuracy(total_acc, class_based_accuracies)
        
        if not isTest:
            update_stats('valid', y, predictions)
        else:
            return update_stats('test', y, predictions)


# # Data structures for overall results

# In[4]:


cumm_train_pred = np.array([])
cumm_train_y = np.array([])
cumm_valid_pred = np.array([])
cumm_valid_y = np.array([])


# In[5]:


def update_stats(dataset, y, predictions):
    global cumm_train_pred, cumm_train_y
    global cumm_valid_pred, cumm_valid_y
    
    preds = np.argmax(predictions, axis=1)
    if dataset == 'train':
        cumm_train_y = np.hstack( (cumm_train_y, y))
        cumm_train_pred = np.hstack( (cumm_train_pred, preds))
    elif dataset == 'valid':
        cumm_valid_y = np.hstack( (cumm_valid_y, y))
        cumm_valid_pred = np.hstack( (cumm_valid_pred, preds))
    else: # test time
        print('=================')
        print('Test Report:')
        print(classification_report(y, preds, target_names=['Class -1', 'Class 1']))
        print('Test Accuracy:', accuracy_score(y, preds))
        
        return preds


# In[6]:


def display_report(X_test, y_test, ann, d):
    global test_accuracies, test_models
    
    print('\n=================')
    print('Overall Train Report:')
    print(classification_report(cumm_train_y, cumm_train_pred, target_names=['Class -1', 'Class 1']))
    print('Overall Train Accuracy:', accuracy_score(cumm_train_y, cumm_train_pred))
    print('=================')
    print('Overall Valid Report:')
    print(classification_report(cumm_valid_y, cumm_valid_pred, target_names=['Class -1', 'Class 1']))
    print('Overall Valid Accuracy:', accuracy_score(cumm_valid_y, cumm_valid_pred))
    print('=================\n')
    
    print("Test Size:",len(X_test))
    a = np.count_nonzero(y_test)
    print("Class Ratios(+1/-1):\t{}/{}".format(a, len(y_test)-a))
    
    return ann.test(X_test, y_test, dropout_keep=d, isTest=True)
    
#     test_accuracies = np.hstack( (test_accuracies, test_acc))
#     test_models = np.hstack( (test_models, ann_instance))


# # Wrapper training class

# In[7]:


# activation is one of relu, tanh, sigmoid (no softmax support for the hidden layers)
def train(epochs=1000, k=10, hidden_size=10, hidden_size2=None, activation='relu', batch_size=128, lr=0.001, d=1):
    
    # read & split data 
    dataset = Dataset('C:/Users/Hassan/Desktop/EE485_data/DATA_Project/dataset.txt')
    skf = dataset.prepare_nn(n_splits=k, normalize=False, shuffle_data=True,                                   oversample=True, undersample=False)
    
    # separate test set (%10) using one of the folds
    X_test = np.array([])
    y_test = np.array([])
    test_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
    for train, test in test_skf.split(dataset.data, dataset.labels):
        dataset.data, X_test = dataset.data[train], dataset.data[test]
        dataset.labels, y_test = dataset.labels[train], dataset.labels[test]
        break
    
    input_size = dataset.data.shape[1]
    output_size = 2
    
    i = 1
    for train_index, test_index in skf.split(dataset.data, dataset.labels):
        X_train, X_valid = dataset.data[train_index], dataset.data[test_index]
        y_train, y_valid = dataset.labels[train_index], dataset.labels[test_index]
        
        print("\nFold {}:\ttrain_size:{} valid_size:{}".format(i, len(X_train), len(X_valid)))
        a = np.count_nonzero(y_train)
        b = np.count_nonzero(y_valid)
        print("Class Ratios(+1/-1):\ttrain:{}/{} valid:{}/{}\n".format(a, len(y_train)-a, b, len(y_valid)-b))
        
        ann = ANN(input_size, output_size, hidden_size=hidden_size, hidden_size2=hidden_size2,                  activation=activation, batch_size=batch_size, lr=lr)
        ann.train(X_train, y_train, epochs, dropout_keep=d)
        ann.test(X_valid, y_valid, dropout_keep=d)
        i += 1
        
        print('----------------------')

    # print overall classification report (test score is based on the last trained ann)
    test_preds = display_report(X_test, y_test, ann, d)
    
    return ann, y_test, test_preds


# In[8]:


# activation is one of relu, tanh, sigmoid (no softmax support for the hidden layers)
# _ = train(epochs=500, k=10, hidden_size=50, hidden_size2=None, activation='relu', batch_size=128, lr=0.001, d=1)


# # Cross-validation

# In[9]:


# cross-validate the hyperparameters using validation set
batch_sizes = [32, 64, 128]
learning_rates = [0.01, 0.001]
activations = ['sigmoid', 'tanh', 'relu']

val_accuracies = np.array([])
val_models = np.array([])
test_accuracies = np.array([])

for bs in batch_sizes:
    for lr in learning_rates:
        for act in activations:
            cumm_train_pred = np.array([])
            cumm_train_y = np.array([])
            cumm_valid_pred = np.array([])
            cumm_valid_y = np.array([])
            
            print('\n\n===========================')
            print('Batch Size:{}, Learning Rate:{}, Activation:{}'.format(bs, lr, act))
            print('===========================\n')
            
            ann, y_test, test_preds = train(epochs=500, k=10, hidden_size=50, hidden_size2=None,                                 activation=act, batch_size=bs, lr=lr, d=1)
            
            val_accuracies = np.hstack( (val_accuracies, accuracy_score(cumm_valid_y, cumm_valid_pred)))
            val_models = np.hstack( (val_models, ann))
            test_accuracies = np.hstack( (test_accuracies, accuracy_score(y_test, test_preds)))

print('\n\n---------RESULTS---------')
t = zip(val_accuracies, val_models, test_accuracies)
sorted_models = sorted(t, key=lambda tup: tup[0], reverse=True)
for m in sorted_models:
    print('Best validation model accuracy:', m[0])
    print('Batch Size:{}, Learning Rate:{}, Activation:{}'.format(m[1].batch_size, m[1].lr, m[1].activation))
    print('Test accuracy of the best model:', m[2])
    break


print('\nOther models:')
i = 0
for m in sorted_models:
    if i == 0:
        i = 1
        continue
    print('Validation model accuracy:', m[0])
    print('Batch Size:{}, Learning Rate:{}, Activation:{}'.format(m[1].batch_size, m[1].lr, m[1].activation))
    print('Test accuracy of the model:', m[2])
    print()


# In[10]:


import datetime
datetime.datetime.now()
