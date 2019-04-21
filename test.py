#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:03:42 2019

@author: anhtu293
"""

#test.py

from skimage import io
from skimage import util
from skimage import transform
from skimage import color
import numpy as np
from skimage.feature import hog

#load model
from sklearn.externals import joblib 
clf = joblib.load('./SVC_RBGF_model.pkl') 

#start Face detection by Sliding Window
stepSize = 10
window_width = 90
window_height = 135
result_detections = np.zeros((0, 6))
c = 0
for k in range(500):
    print(k+1)
    img = io.imread("./test/" + "%04d"%(k+1) +".jpg")
    img = util.img_as_float(img)
    img_pyramid = list()
    img_pyramid.append(img)
    scale = 1
    while img.shape[0]/(2**scale) > window_width and img.shape[1]/(2**scale) > window_height:
        img_pyramid.append(transform.resize(img, (int(np.around(img.shape[0]/(2**scale))), int(np.around(img.shape[1]/(2**scale))))))
        scale += 1
    img_pyramid = tuple(img_pyramid)
    scale = -1
    for img in img_pyramid:
        scale += 1
        img = color.rgb2gray(img)
        for i in range(0,img.shape[0], stepSize):
            for j in range(0, img.shape[1], stepSize):
                window = img[i:min(i+window_height,img.shape[0]), j:min(j+window_width, img.shape[1])]
                window = transform.resize(window,(90,60))
                vecteur = np.zeros((1,3645))
                vecteur[0,:] = hog(window)
                label = clf.predict(vecteur)
                if int(label[0]) == 1:
                    c += 1
                    new_window = np.array([[(k+1), i*(2**scale), j*(2**scale), min(window_height, img.shape[0] - i)*(2**scale), min(window_width, img.shape[1] - j)*(2**scale), clf.decision_function(vecteur)[0]]])
                    result_detections = np.concatenate((result_detections, new_window), axis = 0)
         
#delete doplicated results
res_new = np.zeros((0,6))
for i in range(500):
    result_img = result_detections[result_detections[:,0] == (i+1),:]
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

#output
np.savetxt("./detection.txt", res_new)