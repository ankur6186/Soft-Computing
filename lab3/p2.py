#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:05:17 2019

@author: ankur
"""

import csv
import math
import random
import pandas as pd

def sigmoid(x):
    x = x*-1
    return 1/(1+math.exp(x))

def get(weight,row,i):
    val = 0
    for j in range(len(row)-1):
        val = val+weight[i][j]*row[j]
    return sigmoid(val)

def get1(weight1,l):
    val = 0
    for i in range(len(l)):
        val = val+weight1[i]*l[i]
    return sigmoid(val)

def mlp(train,test,t,learn,count):
    nc = len(train[0])
    weight = []
    weight1 = []
    for i in range(5):
        l = []
        weight1.append(random.uniform(-0.3,0.3))
        for j in range(nc-1):
            l.append(random.uniform(-0.3,0.3))
        weight.append(l)
    for i in range(count):
        l = [0 for k in range(5)]
        for row in train:
            E = [0 for k in range(5)]
            for j in range(5):
                l[j] = get(weight,row,j)
            val = get1(weight1,l)
            error = val*(1-val)*(row[nc-1]-val)
            for j in range(5):
                E[j] = l[j]*(1-l[j])*error*weight1[j]
            for j in range(5):
                weight1[j] = weight1[j]+learn*error*l[j]
            for j in range(5):
                for k in range(nc-1):
                    weight[j][k] = weight[j][k]+learn*E[j]*row[k]
    count = 0.0
    for row in test:
        T = [0 for i in range(5)]
        for j in range(5):
            T[j] = get(weight,row,j)
        val = get1(weight1,l)
        if val >= t:
            val = 1
        else:
            val = 0
        if val == row[nc-1]:
            count = count+1
    return count*100/float(len(test))
            
def fold(dataset,i,k):
    l = len(dataset)
    test_start = int(math.ceil(l/k)*(i-1))
    test_end = int(min(l,math.ceil(l/k)*i))
    if test_start == 0:
        train_start = test_end
        train_end = l
        return [dataset[train_start:train_end],dataset[test_start:test_end]]
    elif test_end == l:
        train_end = test_start
        train_start = 0
        return [dataset[train_start:train_end],dataset[test_start:test_end]]
    else:
        m = []
        for i in range(test_start):
            m.append(dataset[i])
        for i in range(test_end,l):
            m.append(dataset[i])
        return [m,dataset[test_start:test_end]]    

def main():
    dataset = pd.read_csv('SPECT.csv')
    dataset = dataset.sample(frac=1)
    dataset.to_csv('SPECT1.csv')
    filename = "SPECT1.csv"
    dataset = []
    accuracy = []
    
    with open(filename,'r') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for data in rows:
            dataset.append(data)
    
    for i in range(len(dataset)):
        dataset[i].pop(0)

    nr = len(dataset)
    nc = len(dataset[0])
    
    for i in range(nr):
        for j in range(nc):
            if dataset[i][j] == 'Yes':
                dataset[i][j] = 1
            elif dataset[i][j] == 'No':
                dataset[i][j] = 0
            else:
                dataset[i][j] = float(dataset[i][j])
    
    for i in range(nr):
        dataset[i].append(dataset[i][0])
        dataset[i].pop(0)
    
    learn = float(input("Enter the learning rate:        "))
    t = float(input("Enter the threshold:            "))
    count = int(input("Enter the number of iterations: "))
    k = int(input("Enter the number of folds:      "))
    print "\n"
    
    for i in range(1,k+1):
        res = []
        l = fold(dataset,i,k)
        trainset = l[0]
        testset = l[1]
        res = mlp(trainset,testset,t,learn,count)
        accuracy.append(res)
        
    sum = 0.0
    for i in range(len(accuracy)):
        sum = accuracy[i]+sum
    
    print "Accuracy of multiple-layer perceptron is:",sum/len(accuracy),"\n"

if __name__ == '__main__':
    main()