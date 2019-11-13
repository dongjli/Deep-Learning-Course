### EE 526 Homework 1
### Dongjin Li

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from sklearn import preprocessing as pp

############################### Problem 1 ################################

def f(x):
    y = 1 + 2*np.sin(5*x) - np.sin(15*x)
    return y

x = np.linspace(0.0, 1.0, num=51)
xp = np.linspace(0.0, 1.0, num=1000)
def polyplot(x, xp, k, color):
    xmat = np.polynomial.polynomial.polyvander(x, k)
    yp = f(xp)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomials with k=" + str(k))
    for i in range(30):
        y = f(x) + np.random.normal(0, 1, 51)
        beta = np.linalg.inv(xmat.T.dot(xmat)).dot(xmat.T).dot(y)
        yhat = np.poly1d(beta[::-1])(xp)
        plt.plot(xp, yhat, c=color)
    plt.plot(xp, yp, c="r")
    plt.show()


for k in np.arange(1, 12, 2):
    polyplot(x, xp, k, "k")
    plt.savefig("p1_" + str(k) + ".pdf")

############################### Problem 2 ################################

def perceptron(x, y, eta, theta0, eps=1e-6):
    xmat = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    theta1 = theta0
    yhat = np.sign(xmat.dot(theta1))
    theta2 = theta1 + eta * (y - yhat).dot(xmat)
    while np.linalg.norm(theta2 - theta1) > eps:
        theta1 = theta2.copy()
        yhat = np.sign(xmat.dot(theta1))
        theta2 = theta1 + eta * (y - yhat).dot(xmat)
    return theta2

x = np.array([[1, 2], [1, 4], [2, 2], [4, 2], [3, 4], [2, 3]])
y = np.array([1, 1, 1, -1, -1, -1])
theta0 = np.zeros(x.shape[1]+1)
theta = perceptron(x, y, eta=0.1, theta0=theta0)
print(theta)

### Make the plot
plt.figure()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Data Points and Hyperplane")

for yval in np.array([1, -1]):
    ypos = y == yval
    plt.scatter(x[ypos, 0], x[ypos, 1], label="y = " + str(yval))

plt.legend(frameon=True)
x1 = np.linspace(1, 3, num=2)
x2 = (-theta[0] - theta[1]*x1)/theta[2]
plt.plot(x1, x2, c="r")
plt.savefig("p2.pdf")


############################### Problem 3 ################################
### part(a)
dat=pd.read_csv("spambase/spambase.data", header=None)
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def logistic(data, rate, eps=1e-6, max_iter=1e6):
    n_train, p_train = data.shape
    x, y = data[:, :-1], data[:, -1]
    xmat = np.concatenate((np.ones((n_train, 1)), x), axis=1)
    theta2 = np.zeros(p_train)
    pbar = tqdm.tqdm(range(int(max_iter)))
    for i in pbar:
        theta1 = theta2.copy()
        prob = sigmoid(xmat.dot(theta1))
        temp = rate * (prob - y).dot(xmat) / n_train
        theta2 = theta1 - temp
        if np.linalg.norm(temp) < eps * np.linalg.norm(theta1):
            pbar.close()
            break
    return theta2

def lgs_pred(data, theta):
    n_test, p_test = data.shape
    x_test, y_test = data[:, :-1], data[:, -1]
    xmat_test = np.concatenate((np.ones((n_test, 1)), x_test), axis=1)
    yhat = sigmoid(xmat_test.dot(theta))
    err = abs(y_test - (yhat >= 0.5)).mean()
    return err

n, p = dat.shape
dat_spam = dat.loc[dat[p-1] == 1]
dat_ham = dat.loc[dat[p-1] == 0]
spam_cut = len(dat_spam)//3*2
ham_cut = len(dat_ham)//3*2

### unnormalized data
train_unscaled = np.vstack((dat_spam[0:spam_cut], dat_ham[0:ham_cut]))
test_unscaled = np.vstack((dat_spam[spam_cut:n], dat_ham[ham_cut:n]))
errors_unscaled = []
theta_unscaled = []
for rate in np.array([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]):
    theta = logistic(train_unscaled, rate)
    theta_unscaled.append(theta)
    err_unscaled = lgs_pred(test_unscaled, theta)
    errors_unscaled.append(err_unscaled)
print(errors_unscaled)

### normalized data
train_scaled = train_unscaled.copy()
test_scaled = test_unscaled.copy()
train_scaled[:, :-1] = pp.scale(train_unscaled[:, :-1])
test_scaled[:, :-1] = pp.scale(test_unscaled[:, :-1])
errors_scaled = []
theta_scaled = []
for rate in np.array([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]):
    theta = logistic(train_scaled, rate)
    theta_scaled.append(theta)
    err_scaled = lgs_pred(test_scaled, theta)
    errors_scaled.append(err_scaled)
print(errors_scaled)