### EE 526 Homework 2
### Dongjin Li
from DNN import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing as pp

############################### Problem 5 ################################
dat=pd.read_csv("spambase/spambase.data", header=None)
train_scaled = train_unscaled.copy()
test_scaled = test_unscaled.copy()
train_scaled[:, :-1] = pp.scale(train_unscaled[:, :-1])
test_scaled[:, :-1] = pp.scale(test_unscaled[:, :-1])

def plot(X, Y, nn):
  plt.ion()
  fig=plt.figure(1)
  plt.clf()
  x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
  y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)

  plt.scatter(X[0, :], X[1, :], c=np.argmax(Y,axis=0),
    s=40)

  if nn is not None:
    N=150
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
      np.linspace(y_min, y_max, N))
    Z = nn.predict( np.c_[xx.ravel(), yy.ravel()].T )
    Z = np.argmax(Z, axis=0)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,  alpha=0.3)
    fig.canvas.start_event_loop(0.01)

def main():
  D=57 # Input dimension
  Odim=1 # number of outputs
  layers=[ (100, ReLU), (40, ReLU), (Odim, ) ]
  nn=NeuralNetwork(D, layers)
  nn.setRandomWeights(0.1)
  CE=ObjectiveFunction('crossEntropyLogit')

  N=200 # points per cluster
  K=3 # number of clusters
  X,Y=generateData(N,K,D)

  eta=1e-1

  for i in range(10000):
    logp=nn.doForward(X)
    J=CE.doForward(logp, Y)
    dz=CE.doBackward(Y)
    dx=nn.doBackward(dz)
    nn.updateWeights(eta)
    if (i%100==0):
      print( '\riter %d, J=%f' % (i, J), end='')
      plot(X,Y,nn)
    # nn.print(['W', 'b'])
  input('Press Enter to Finish')

main()

# run: clear; python %
# run: PYTHONBREAKPOINT=pudb.set_trace python -Werror -m pudb.run %
