import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib.pyplot as plt
from sklearn import metrics

# Reading in the training variables
Y_train = pd.read_csv('Y_madelon_train.txt', delim_whitespace=True, header=None)
X_train = pd.read_csv('X_madelon_train.txt', delim_whitespace=True, header=None)

# Reading in the testing variables
Y_valid = pd.read_csv('Y_madelon_valid.txt', delim_whitespace=True, header=None)
X_valid = pd.read_csv('X_madelon_valid.txt', delim_whitespace=True, header=None)

# Creating 'k' to set the number of trees needed
k = np.array([3, 10, 30, 100, 300])

training_error = np.zeros((1,5))
testing_error = np.zeros((1,5))

# PART(A): Creating the random forest for square root 500
for x in range(0,5):
    train_rnd_clf1 = RandomForestClassifier(n_estimators=k[x], max_features=int(np.sqrt(500)),)
    train_rnd_clf1.fit(X_train, Y_train)
    Y_predict_train = train_rnd_clf1.predict(X_train)
    Y_predict_valid = train_rnd_clf1.predict(X_valid)
# Calculating training error
    training_error[0,x] = np.mean(Y_predict_train != np.array(Y_train).reshape(1,-1))
# Calculating testing error
    testing_error[0,x] = np.mean(Y_predict_valid != np.array(Y_valid).reshape(1,-1))
# plot
plt.plot(k, training_error.T)
plt.plot(k, testing_error.T)
plt.xlabel('n_trees')
plt.ylabel('Misclassification Error')
plt.legend(('Train','Valid'))
plt.show

    
# PART(B): Creating the random forest for log 500
for x in range(0,5):
    train_rnd_clf1 = RandomForestClassifier(n_estimators=k[x], max_features=int(np.log(500)),)
    train_rnd_clf1.fit(X_train, Y_train)
    Y_predict_train = train_rnd_clf1.predict(X_train)
    Y_predict_valid = train_rnd_clf1.predict(X_valid)
# Calculating training error
    training_error[0,x] = np.mean(Y_predict_train != np.array(Y_train).reshape(1,-1))
# Calculating testing error
    testing_error[0,x] = np.mean(Y_predict_valid != np.array(Y_valid).reshape(1,-1))
# plot
plt.plot(k, training_error.T)
plt.plot(k, testing_error.T)
plt.xlabel('n_trees')
plt.ylabel('Misclassification Error')
plt.legend(('Train','Valid'))
plt.show    

# PART(C): Creating the random forest for 500
for x in range(0,5):
    train_rnd_clf1 = RandomForestClassifier(n_estimators=k[x], max_features=None,)
    train_rnd_clf1.fit(X_train, Y_train)
    Y_predict_train = train_rnd_clf1.predict(X_train)
    Y_predict_valid = train_rnd_clf1.predict(X_valid)
# Calculating training error
    training_error[0,x] = np.mean(Y_predict_train != np.array(Y_train).reshape(1,-1))
# Calculating testing error
    testing_error[0,x] = np.mean(Y_predict_valid != np.array(Y_valid).reshape(1,-1))
# plot
plt.plot(k, training_error.T)
plt.plot(k, testing_error.T)
plt.xlabel('n_trees')
plt.ylabel('Misclassification Error')
plt.legend(('Train','Valid'))
plt.show 


