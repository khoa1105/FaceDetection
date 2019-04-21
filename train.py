#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:53:27 2019

@author: anhtu293
"""
#train.py
#Create model

from skimage import io
from skimage import util
from skimage import transform
from skimage import color
import numpy as np
from skimage.feature import hog

#load train set
label = np.loadtxt('./label.txt', delimiter = ' ')

largeur = 60
hauteur = 90
hog_dim = 3645
train_pos = np.zeros((label.shape[0], hog_dim))
train_neg = np.zeros((5000, hog_dim))
y_train_pos = np.ones((label.shape[0],1))

#Find all faces from train set
for i in range(label.shape[0]):
    image = io.imread("./train/" + "%04d"%(label[i,0]) + ".jpg")
    image = util.img_as_float(image)
    image = image[int(label[i,1]):int(label[i,1]+label[i,3]),int(label[i,2]):int(label[i,2]+label[i,4])]
    image = color.rgb2gray(image)
    image = transform.resize(image,(hauteur, largeur))
    train_pos[i,:] = hog(image)
    
#create random negative exemples 
y_train_neg = np.ones((5000,1))
for i in range(1000):
    image = io.imread("./train/" + "%04d"%(i+1) + ".jpg")
    image = util.img_as_float(image)
    for j in range(5):
        ligne = np.random.randint(0, image.shape[0]-10)
        colonne = np.random.randint(0, image.shape[1]-10)
        hauteur_rand = np.random.randint(40, image.shape[0])
        largeur_rand = np.random.randint(40, image.shape[1])
        image = image[ligne:min(ligne+hauteur_rand,image.shape[0]-1),colonne:min(colonne+largeur_rand, image.shape[1]-1)]
        image = color.rgb2gray(image)
        image = transform.resize(image,(hauteur, largeur))
        train_neg[i*5+j,:] = hog(image)
                
#add more negative exemples from False Positive of previous classifier 
res_new = np.loadtxt("./res_new.txt", delimiter = ' ')
train_neg_sup = np.zeros((res_new.shape[0], hog_dim))
neg_sup = res_new[res_new[:,6] == 0,0:5]
neg_sup = neg_sup.astype(int)
for i in range(neg_sup.shape[0]):
    image = io.imread("./train/" + "%04d"%(neg_sup[i,0]) + ".jpg")
    image = util.img_as_float(image)
    image = color.rgb2gray(image)
    image = image[(neg_sup[i,1]):(neg_sup[i,1]+neg_sup[i,3]),(neg_sup[i,2]):(neg_sup[i,2]+neg_sup[i,4])]
    image = transform.resize(image,(hauteur, largeur))
    train_neg_sup[i,:] = hog(image)
train_neg = np.concatenate((train_neg, train_neg_sup), axis = 0)

res_new = np.loadtxt("./res_new2.txt", delimiter = ' ')
train_neg_sup = np.zeros((res_new.shape[0], hog_dim))
neg_sup = res_new[res_new[:,6] == 0,0:5]
neg_sup = neg_sup.astype(int)
for i in range(neg_sup.shape[0]):
    image = io.imread("./train/" + "%04d"%(neg_sup[i,0]) + ".jpg")
    image = util.img_as_float(image)
    image = color.rgb2gray(image)
    image = image[(neg_sup[i,1]):(neg_sup[i,1]+neg_sup[i,3]),(neg_sup[i,2]):(neg_sup[i,2]+neg_sup[i,4])]
    image = transform.resize(image,(hauteur, largeur))
    train_neg_sup[i,:] = hog(image)
train_neg = np.concatenate((train_neg, train_neg_sup), axis = 0)

#save negative training set
np.savetxt("./train_neg.txt", train_neg)

#concatenate positive and negative training set
train = np.concatenate((train_pos, train_neg), axis = 0)
y_train = np.concatenate((np.ones(train_pos.shape[0]), -np.ones(train_neg.shape[0])))

#create model
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', gamma = 'scale')
clf.fit(train, y_train)

#save model
from sklearn.externals import joblib 
joblib.dump(clf, './SVC_RBGF_model.pkl')