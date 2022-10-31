# This program contain multiple Deep Learning methods to import in other programs.
# Author: Adrien Dorise
# Date: January 2021

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import os
import math

#Importing torch library for RBM and AE
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importing the Keras libraries and packages for RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras import models

#Anomaly detection with Auto Encoders
# https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

#Utilisation GPU
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)







#batchCreation divide a single set into multiple batch. This is usefull when you want to do the training on a single data set
def batchCreation(dataSet, batchSize):
    #PROBLEM: BATCh size takes the lowest round value. Meaning we are loosing last data points of data set
    print("TODO: Batch size takes the lowest round value. Meaning we are loosing last data points of data set")
    print("TODO: bath only take 1D input")
    batchedDataSet = []
    tempData = []
    nbBatch = int(len(dataSet)/batchSize)
    for i in range(0,nbBatch*batchSize,batchSize):
        batchedDataSet.append(list(dataSet[i:i+batchSize,0]))
    return batchedDataSet
        

def debatching(batchDataSet, batchSize):
    dataSet = []
    print("Batch size " + str(len(batchDataSet)))
    for batch in range(len(batchDataSet)):
        for i in range(batchSize-1):
            dataSet.append(batchDataSet[batch][0,i])
    return np.array(dataSet).reshape(-1,1)






#Restricted Boltzmann Machines (Class; train; test; print)
class RBM():
    def __init__(self, nv, nh): #nv is number of visible nodes and nh is number of hidden nodes
        self.W = torch.randn(nh, nv) #Weights of visible nodes according to hidden nodes
        self.a = torch.randn(1, nh) #bias for probability of hidden nodes given visible nodes: PH given V (first dimension is batch dimension)
        self.b = torch.randn(1, nv) #bias for probability of visible nodes given hidden nodes: PV given H (first dimension is batch dimension)
    def sample_h(self, x): #Hidden nodes activation
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx) #Activation of hidden nodes given proba of visible nodes
        p_h_given_v = torch.sigmoid(activation) #We compute activation proba in sigmoid function
        return p_h_given_v, torch.bernoulli(p_h_given_v) #We return the proba of activation + the activated sampled hidden neurons given the proba
    def sample_v(self, y): #Visible nodes activation
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk): #Using of contrastive divergence in Gibbs sampling
        # self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.W += (torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0) #Updating of the bias b
        self.a += torch.sum((ph0 - phk), 0) #Updating of the bias a
    
    
def RBMtrain(batchedDataScaled, hiddenNodes = 200, nbEpochs =150,random_state = None):
    inputSize = len(batchedDataScaled[0])
    batchSize = len(batchedDataScaled)
    rbm = RBM(inputSize, hiddenNodes)    
    batchedDataScaled = torch.FloatTensor(batchedDataScaled)

    #Training
    for epoch in range(1, nbEpochs + 1):
        train_loss = 0
        s = 0.
        inputList = []
        outputList = []
        for batch in range(batchSize):
            vk = batchedDataScaled[batch:batch+1] #state after k iterations
            v0 = batchedDataScaled[batch:batch+1] #initial state
            ph0,_ = rbm.sample_h(v0)
            for k in range(10): #We do the random walk of Gibbs sampling
                _,hk = rbm.sample_h(vk) #We recreate hidden nodes
                _,vk = rbm.sample_v(hk) #Then we recreate visible nodes
            phk,_ = rbm.sample_h(vk) 
            rbm.train(v0, vk, ph0, phk) #Training
            train_loss += torch.mean(torch.abs(v0 - vk)) #Loss calculation
            s += 1.
            inputList.append(v0.data.numpy())
            outputList.append(vk.data.numpy())
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
    return rbm,inputList,outputList



def RBMtest(batchedDataScaled, regressor):
    # Testing the SAE
    batchSize = len(batchedDataScaled)
    batchedDataScaled = torch.FloatTensor(batchedDataScaled)
    test_loss = 0
    s = 0.
    inputList = []
    outputList = []
    for batch in range(batchSize):
        v = batchedDataScaled[batch:batch+1] #
        vt = batchedDataScaled[batch:batch+1]
        if len(vt[vt>=0]) > 0:
            #We do one step of creating hidden and visible nodes to get the prediction
            _,h = regressor.sample_h(v) 
            _,v = regressor.sample_v(h)
            test_loss += torch.mean(torch.abs(vt - v))
            s += 1.
        inputList.append(vt.data.numpy())
        outputList.append(v.data.numpy())
    print('test loss: '+str(test_loss/s))
    return inputList,outputList


def printRBM(realData,prediction):
    #Difference between real and prediciton
    diff=[]
    for i in range(len(prediction)):
        diff.append(math.sqrt(math.pow(prediction[i,0]-realData[i],2)))
    
    # Visualising the results
    plt.plot(realData, color = 'green', label = 'Real Data')
    plt.plot(prediction, color = 'blue', label = 'Predicted Data')
    plt.title('Current prediction using SAE')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.show()
    plt.plot(diff, color = 'blue', label = 'Root Square Distance')
    plt.title("Difference between target and prediction")
    plt.show()






#Auto Encoders (Class; train; test; print)
#With AE, we have to partitionate the dataset, as we will learn features on each section of the dataset.
#The input correspond to a certain behaviour/mode
# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, inputSize, layersArchitecture):
        super(SAE, self).__init__()
        self.inputSize = inputSize
        self.activation = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        
        fc = []  #We use an array to store all layers of the network
        fc.append(nn.Linear(inputSize, layersArchitecture[0]))
        for layer in range(len(layersArchitecture)):
            if layer == len(layersArchitecture)-1:
                #Output layer
                fc.append(nn.Linear(layersArchitecture[layer], inputSize))
                print("Output")
            else:
                #Hidden layers
                fc.append(nn.Linear(layersArchitecture[layer], layersArchitecture[layer+1]))
                print("Hidden")
            print(layersArchitecture[layer])
            print(fc[layer])
        self.fc = nn.ModuleList(fc) #We have to use ModuleList instead of normal Python list to still access pytorch functionality
        print(self.fc)
        self.optimizer = optim.RMSprop(self.parameters(), lr = 0.01, weight_decay = 0.5)
    def forward(self, x):
        #x corresponds to the infos going through the network.
        for layer in range(len(self.fc)):
            if layer < len(self.fc)-1:
            #Only input + hidden layers
                x = self.activation(self.fc[layer](x))

            else:
            #Output layer
                x = self.fc[layer](x)
        return x



#batchSize correspond to the number of points inside a batch
def SAEtrain(batchedDataScaled,layersArchitecture = [80,50], nbEpochs =150,random_state = None):
    inputSize = len(batchedDataScaled[0])
    batchSize = len(batchedDataScaled)
    sae = SAE(inputSize, layersArchitecture)    
    batchedDataScaled = torch.FloatTensor(batchedDataScaled)

    #Training
    for epoch in range(1, nbEpochs + 1):
        train_loss = 0
        s = 0.
        inputList = []
        outputList = []
        for batch in range(batchSize):
            input = Variable(batchedDataScaled[batch]).unsqueeze(0)
            target = input.clone()
            output = sae.forward(input)
            target.require_grad = False #We make sure to not compute the gradient with the target, only the outputs during backpropagation (by default in Pytorch, the require_grad is set to true)
            loss = sae.criterion(output, target)
            loss.backward()
            train_loss += np.sqrt(loss.data)
            s += 1.
            sae.optimizer.step()
            # print("Value batch = "+ str(batch))
            inputList.append(input.data.numpy())
            outputList.append(output.data.numpy())
        #The loss represnet the difference between real values and prediction for a single prediction
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
    return sae,inputList,outputList



def SAEtest(batchedDataScaled, regressor):
    # Testing the SAE
    batchSize = len(batchedDataScaled)
    batchedDataScaled = torch.FloatTensor(batchedDataScaled)
    test_loss = 0
    s = 0.
    inputList = []
    outputList = []
    for batch in range(batchSize):
        input = Variable(batchedDataScaled[batch]).unsqueeze(0)
        target = input.clone()
        output = regressor(input)
        target.require_grad = False
        loss = regressor.criterion(output, target)
        test_loss += np.sqrt(loss.data)
        s += 1.
        inputList.append(input.data.numpy())
        outputList.append(output.data.numpy())
    print('test loss: '+str(test_loss/s))
    return inputList,outputList


def printSAE(realData,prediction):
    #Difference between real and prediciton
    diff=[]
    predTemp=[]
    for i in range(len(prediction)):
        diff.append(math.sqrt(math.pow(prediction[i][0][0]-realData[i,0],2)))
        predTemp.append(prediction[i][0][0])
    # Visualising the results
    plt.plot(realData, color = 'green', label = 'Real Data')
    plt.plot(predTemp, color = 'blue', label = 'Predicted Data')
    plt.title('Current prediction using SAE')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.show()
    plt.plot(diff, color = 'blue', label = 'Root Square Distance')
    plt.title("Difference between target and prediction")
    plt.show()
    

#RNN (TRAIN; TEST; PRINT)
#For RNN, we can give the whole dataset at once, as it is trying to predict the next value.
def RNNtrain(trainDataScaled, layersArchitecture = [100,90,90,80], timeSteps = 300, predictionSteps = 20, nbEpochs = 150, batch_size=320, random_state=None):
    # Creating a data structure with X timeSteps and 1 output
    X_train = []
    y_train = []
    for i in range(timeSteps, len(trainDataScaled)-predictionSteps):
        X_train.append(trainDataScaled[i-timeSteps:i,0])
        y_train.append(trainDataScaled[i+predictionSteps,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Building the RNN 
    layer=0
    # Initialising the RNN
    regressor = Sequential()
    
    return_sequences = True
    for layer in range(len(layersArchitecture)+1):
        print("layer " + str(layer))
        if layer >= len(layersArchitecture)-1:
            #In case it is the last layer before output
            print("Return Sequence FALSE")
            return_sequences = False
            
        if layer == 0:
            print("Init " + str(layersArchitecture[layer]))
            # Adding the first LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units = layersArchitecture[layer], return_sequences = return_sequences, input_shape = (X_train.shape[1], 1)))
            regressor.add(Dropout(0.2))
            
        elif layer <= len(layersArchitecture)-1:
            print("add layer " + str(layersArchitecture[layer]))
             # Adding hidden LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units = layersArchitecture[layer], return_sequences = return_sequences))
            
        elif layer == len(layersArchitecture):
            print("output " + str(1))
            # Adding the output layer
            regressor.add(Dense(units = 1))
        else:
            print("WARNING IN RNNTRAIN: Unknown error in network initialisation. Code error 1")
            return 1;
        

    # Compiling the RNN
    # HERE YOU HAVE TO CHOOSE THE OPTIMISER WANTED. RMS IS ALSO USUALLY A GOOD CHOICE.
    #ALSO CHOOSE THE LOSS VALUE CALCULATION
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    history = regressor.fit(X_train, y_train, epochs = nbEpochs, batch_size = batch_size)
    loss = history.history['loss']

    return regressor, loss
   
    
'''
Note that the first values of prediction corresponds to the timeSteps needed to the training
'''
def RNNtest(testDataScaled, regressor, timeSteps):
    

    #Prediction
    
    #Test with no latch
    # dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    # inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = testDataScaled[:,0:1]
    inputs = inputs.reshape(-1,1)
    X_test = []
    for i in range(timeSteps, len(inputs)):
        X_test.append(inputs[i-timeSteps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    prediction = regressor.predict(X_test) #Prediction of all timeSteps
    return prediction
    
    

def printPredictionRNN(realData, prediction, timeSteps, predictionSteps):
    #Difference between real and prediciton
    diff=[]
    for i in range(len(prediction)-predictionSteps):
        diff.append(math.sqrt(math.pow(prediction[i,0]-realData[i+timeSteps+predictionSteps],2)))
    
    # Visualising the results
    plt.plot(realData[timeSteps+predictionSteps:len(realData)], color = 'green', label = 'Real Data')
    plt.plot(prediction[:,0], color = 'blue', label = 'Predicted Data')
    # plt.plot(prediction[:,len(prediction[0,:])-1], color = 'blue', label = 'Predicted Data')
    plt.title('Current prediction using RNN')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.show()
    plt.plot(diff, color = 'blue', label = 'Root Square Distance')
    plt.title(" target and prediction")
    plt.show()
    return diff



# One Class Classification with Auto Encoders
class AutoEncoder(Model):
  """
  Parameters
  ----------
  output_units: int
    Number of output units
  
  code_size: int
    Number of units in bottle neck
  """

  def __init__(self, output_units, code_size=4):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

def find_threshold(model, x_train_scaled):
    reconstructions = model.predict(x_train_scaled)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) \
        + np.std(reconstruction_errors.numpy())
    return threshold

def get_predictions(model, x_test_scaled, threshold):
    predictions = model.predict(x_test_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) <= threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds


def trainAE(trainSetScaled, epochs = 20, batch_size=512):
    x_train_scaled = trainSetScaled


    model = AutoEncoder(output_units=x_train_scaled.shape[1])
    # configurations of model
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')
    
    history = model.fit(
        x_train_scaled,
        x_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        # validation_data=(x_test_scaled, trainSetScaled)
    )
    threshold = find_threshold(model, x_train_scaled)
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSLE Loss')
    plt.legend(['loss', 'val_loss'])
    plt.show()
    return model, threshold

def predictAE(model,threshold,testSetScaled):

    predictions = get_predictions(model, testSetScaled, threshold)
    
    # accuracy = accuracy_score(predictions, y_test)
    return predictions

def confusionMatrixAE(prediction,realClass,faultValue):
    confusionMatrix = [0,0,0,0]
    for i in range(len(prediction)):
        # if prediction[i] == 1 and realClass[i] == True:
        #     confusionMatrix = np.add(confusionMatrix,[1,0,0,0])
        # elif prediction[i] == 1 and realClass[i] == False:
        #     confusionMatrix = np.add(confusionMatrix,[0,1,0,0])
        # elif prediction[i] == 0 and realClass[i] == True:
        #     confusionMatrix = np.add(confusionMatrix,[0,0,1,0])
        # elif prediction[i] == 0 and realClass[i] == False:
        #     confusionMatrix = np.add(confusionMatrix,[0,0,0,1])
            
        #TP FP FN TN
        if prediction[i] == 0 and (realClass[i] != faultValue and realClass[i] != 100):
            confusionMatrix = np.add(confusionMatrix,[0,0,0,1])
        elif prediction[i] == 0 and realClass[i] == faultValue:
            confusionMatrix = np.add(confusionMatrix,[0,0,1,0])
        elif prediction[i] == 1 and realClass[i] != faultValue:
            confusionMatrix = np.add(confusionMatrix,[0,1,0,0])
        elif (prediction[i] == 1 and realClass[i] == faultValue) or realClass[i] == 100:
            confusionMatrix = np.add(confusionMatrix,[1,0,0,0])
        
    return confusionMatrix


def saveModel(model,path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+".h5")
    print("Saved model to disk")
    
    
    
def loadModel(path):    
    #Load a model
    # load json and create model
    json_file = open(path+".json", 'r')
    loadedModel = json_file.read()
    json_file.close()
    loadedModel = models.model_from_json(loadedModel)
    # load weights into new model
    loadedModel.load_weights(path+".h5")
    print("Loaded model from disk")
    return loadedModel

    

