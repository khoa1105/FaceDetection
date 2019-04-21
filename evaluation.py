#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:14:05 2019

@author: anhtu293
"""

import numpy as np
import matplotlib.pyplot as plt


res_new = np.zeros((0,6))
result = np.loadtxt("./result_validation.txt")
for i in range(500,1000,1):
    result_img = result[result[:,0] == (i+1),:]
    for k in range(result_img.shape[0]):
        boite1 = result_img[k,:]
        if boite1[0] == -1:
            continue
        for j in range(result_img.shape[0]):
            if j == k or result_img[j,0] == -1:
                continue
            boite2 = result_img[j,:]
            grille = np.zeros((max(int(boite1[1]) + int(boite1[3]), int(boite2[1]) + int(boite2[3])), max(int(boite1[2]) + int(boite1[4]), int(boite2[2]) + int(boite2[4]))))
            grille[int(boite1[1]):(int(boite1[1]) + int(boite1[3])), int(boite1[2]):(int(boite1[2]) + int(boite1[4]))] += 1
            grille[int(boite2[1]):(int(boite2[1]) + int(boite2[3])), int(boite2[2]):(int(boite2[2]) + int(boite2[4]))] += 1
            aire_totale = np.size(np.ravel(grille)[np.ravel(grille)[:] > 0])
            aire_intersection = np.size(np.ravel(grille)[np.ravel(grille)[:] == 2])
            if (float)(aire_intersection/aire_totale) >= 0.5:
                if boite1[5] > boite2[5]:
                    result_img[j,0] = -1
                else:
                    result_img[k,0] = -1
                    break
    res_new = np.concatenate((res_new, result_img[result_img[:,0] != -1,:]), axis = 0)

#True positive & False positive
label = np.loadtxt('./label.txt', delimiter = ' ')   
TrueFalse_positive = np.zeros((res_new.shape[0],1))
label_validation = label[label[:,0] > 500, :]
label_validation = np.concatenate((label_validation, np.zeros((label_validation.shape[0], 1))), axis = 1)
label_validation = label_validation.astype(int)

for i in range(res_new.shape[0]):
    boite1 = res_new[i, :]
    for j in range(label_validation.shape[0]):
        if label_validation[j,0] != int(boite1[0]):
            continue
        boite2 = label_validation[j,:]
        grille = np.zeros((max(int(boite1[1]) + int(boite1[3]), int(boite2[1]) + int(boite2[3])), max(int(boite1[2]) + int(boite1[4]), int(boite2[2]) + int(boite2[4]))))
        grille[int(boite1[1]):(int(boite1[1]) + int(boite1[3])), int(boite1[2]):(int(boite1[2]) + int(boite1[4]))] += 1
        grille[int(boite2[1]):(int(boite2[1]) + int(boite2[3])), int(boite2[2]):(int(boite2[2]) + int(boite2[4]))] += 1
        aire_totale = np.size(np.ravel(grille)[np.ravel(grille)[:] > 0])
        aire_intersection = np.size(np.ravel(grille)[np.ravel(grille)[:] == 2])
        if (float)(aire_intersection/aire_totale) >= 0.5:
            TrueFalse_positive[i,0] = 1
            label_validation[j, 5] = 1
            break
        
#False negative
label_validation = label[label[:,0] > 500, :]
label_validation = label_validation.astype(int)
False_negative = np.zeros((label_validation.shape[0],1))

for i in range(label_validation.shape[0]):
    boite1 = label_validation[i,:]
    for j in range(res_new.shape[0]):
        if boite1[0] != int(res_new[j,0]):
            continue
        boite2 = res_new[j, :]
        grille = np.zeros((max(int(boite1[1]) + int(boite1[3]), int(boite2[1]) + int(boite2[3])), max(int(boite1[2]) + int(boite1[4]), int(boite2[2]) + int(boite2[4]))))
        grille[int(boite1[1]):(int(boite1[1]) + int(boite1[3])), int(boite1[2]):(int(boite1[2]) + int(boite1[4]))] += 1
        grille[int(boite2[1]):(int(boite2[1]) + int(boite2[3])), int(boite2[2]):(int(boite2[2]) + int(boite2[4]))] += 1
        aire_totale = np.size(np.ravel(grille)[np.ravel(grille)[:] > 0])
        aire_intersection = np.size(np.ravel(grille)[np.ravel(grille)[:] == 2])
        if (float)(aire_intersection/aire_totale) >= 0.5:
            False_negative[i,0] = 1
            break

#Precision, Recall and F1

    #sort predictions by score
res_new =  np.concatenate((res_new, TrueFalse_positive), axis = 1)
predictions = res_new[np.argsort(res_new[:,5]),:]
predictions = predictions[-1:-(res_new.shape[0]+1):-1,:]

    #calcul Precision P Recall R score F1
precision = np.zeros((predictions.shape[0],1))
recall = np.zeros((predictions.shape[0],1))
F1 = np.zeros((predictions.shape[0],1))

n_groundtruth = False_negative.shape[0]
TP = 0
FP = 0
for i in range(predictions.shape[0]):
    if int(predictions[i,6]) == 1:
        TP += 1
    else:
        FP += 1
    precision[i,0] = (float)(TP/(TP+FP))
    recall[i,0] = (float)(TP/(n_groundtruth))
    if recall[i,0] == 0 or precision[i,0] == 0:
        F1[i,0] = 0
    else:
        F1[i,0] = (float)((2*precision[i,0]*recall[i,0])/(recall[i,0] + precision[i,0]))

plt.plot(recall, precision, linewidth=2.0)

from sklearn.metrics import auc
auc = auc(recall, precision)

    







