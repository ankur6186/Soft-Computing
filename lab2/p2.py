#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:36:43 2019

@author: ankur
"""

import csv
import math
import random
import pandas as pd

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

def get(weight,row,nc,t):
    val = 0.0
    for i in range(nc-1):
        val = val+weight[i]*row[i]
    if val >= t:
        return 1
    return 0

def perceptron(train,test,t,learn,count):
    weight = []
    nc = len(train[0])
    for i in range(nc-1):
        weight.append(random.uniform(0,1))
    for i in range(count):
        for row in train:
            val = get(weight,row,nc,t)
            error = row[nc-1]-val
            for j in range(nc-1):
                weight[j] = weight[j]+learn*error*row[j]
    
    count = 0.0
    for row in test:
        val = get(weight,row,nc,t)
        if val == row[nc-1]:
            count = count+1
    return (count*100)/float(len(test))

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
            if dataset[i][j] == 'No':
                dataset[i][j] = 0
            elif dataset[i][j] == 'Yes':
                dataset[i][j] = 1
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
        l = fold(dataset,i,k)
        trainset = l[0]
        testset = l[1]
        res = perceptron(trainset,testset,t,learn,count)
        accuracy.append(res)
    
    sum = 0.0
    for i in range(len(accuracy)):
        sum = accuracy[i]+sum
        print "Accuracy of",i,"-th fold is:",accuracy[i],"\n"
    
    print "Accuracy of single-layer perceptron is:",sum/len(accuracy),"\n"
    
if __name__ == '__main__':
    main()