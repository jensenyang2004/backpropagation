# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:28:40 2022

@author: user
"""
import numpy as np
import copy
# Third-party libraries
class ann:
    def __init__(self,layers):
        #[4,13,14,15,4]
        self.layers = layers
        self.nums = len(layers)
        self.result = []
        self.weight = [np.random.randn(x,y) for x,y in zip(layers[1:],layers)]
        self.bias = [np.random.randn(p) for p in layers[1:]]
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_der(z):
    a = sigmoid(z)
    return a*(1-a)
def feedforward(ann,input_):
    input_data = []
    train_dataZ = []
    input_data.append(input_)
    for i in range (ann.nums-1): 
        z = np.matmul(ann.weight[i],input_data[i])
        z += ann.bias[i]
        temp = copy.copy(z)
        train_dataZ.append(temp)
        for j in range(len(z)):
           z[j]=sigmoid(z[j])
        input_data.append(z)  
    ann.result = input_data[ann.nums-1]
    print(ann.result)
    return train_dataZ,input_data
def backpropagation(ann,train_data,correction,input_data,input_v):
    #first derivative setting
    C0_deri = ann.result-correction
    C0_deri = C0_deri*2
    for i in range(ann.nums-2,-1,-1):
        #step 1: sigmoid derivative
        sigmoid = [sigmoid_der(z) for z in train_data[i]]
        #step 2: weight derivative
        weight = [input_data[i] for x in range(len(ann.weight[i]))]
        a_deriva = []
        #dC0/da*da/dz
        sigmoid = [x*y for x,y in zip(sigmoid,C0_deri)]
        #derivative of a
        for k in range(ann.layers[i]):
            sum = 0
            for j in range(ann.layers[i+1]):
                sum += ann.weight[i][j][k]*sigmoid[j]
            a_deriva.append(sum)
        #calculation has been completed
        for j in range (len(ann.weight[i][0])):
            #n
            for k in range(len(ann.weight[i])):
                #m
                weight[k][j]=weight[k][j]*sigmoid[k]
        learning(i, weight, sigmoid, a_deriva,ann,input_v)
        C0_deri = copy.copy(a_deriva)
    return 0  
def learning(i,weight,bias,a_deriva,ann,input_v):
    learning_rate = 10
    delta_bias = [-1*learning_rate*x for x in bias]
    delta_weight=[]
    for j in range(len(weight)):
        temp = [-1*learning_rate*x for x in weight[j]]
        delta_weight.append(temp)
    #print(delta_weight)
    ann.weight[i]+=delta_weight
    ann.bias[i]+=delta_bias
    feedforward(ann, input_v)
network = ann([9,16,9])
while(True):
    input_v = input("Enter dataset").split()
    input_v = [float(x) for x in input_v]
    train_data,input_data = feedforward(network,input_v)
    correction = input("Enter correction ").split()
    correction = [float(x) for x in correction]
    backpropagation(network,train_data,correction,input_data,input_v)

    