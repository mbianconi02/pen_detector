# Pen Identifier
# The goal of this script is to implement a classifier which is able to determine the tip and the end of a pen
# This was one of the first ML assignments I performed and although simple I enjoyed doing it
# DISCLAIMER: this originally was a Jupyter notebook with interactive widgets but now has been converted to a simple python script
# Hence there may be some small bugs that need fixing which haven't been thoroughly checked

#imports

import sys
import sklearn
import numpy as np
import os, glob

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.close('all')

import scipy

import skimage
import skimage.transform
import skimage.util

import pickle

from pen_functions import *
       

#General terms
display_images = True #If True it will display intermediate steps of what the code does
save_model = False #If True saved all classifier models trained
img_idx = 0

"""Load Data"""

###Load Images###

IMAGE_DIR = 'images/mypen'

filenames = list_images(IMAGE_DIR)
N = len(filenames)

Is = [plt.imread(filename) for filename in filenames]
print('loaded %d images' % len(Is))

# Display first image
if display_images:
    
    plt.figure()
    plt.imshow(Is[img_idx])
    plt.show()

###Load annotations (labels)###
annot_filename = os.path.join(IMAGE_DIR, 'annots.npy')
annots = pickle.load(open(annot_filename, 'rb'))

# Display image including annotations
I = Is[img_idx]
p1 = annots[img_idx,:2].copy() # point 1, tip of the pen
p2 = annots[img_idx,2:].copy() # point 2, end of the pen

if display_images:
    show_annotation(I, p1, p2)

"""Extract patches from images"""

# Define patch size
WIN_SIZE = (100, 100, 3)
HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])

P = get_patch_at_point(I, p1, WIN_SIZE)
if display_images:
    plt.imshow(P)
    plt.show()

# Transform patch to feature vector

FEAT_SIZE = (9,9,3)

number_of_feature_dimensions = np.int64(patch_to_vec(P, FEAT_SIZE).shape[0])
print(f'This will be a {number_of_feature_dimensions}-dimensional feature space')

# Sample locations in image to extract patches

#This is done with 2 startegies
#1. sample in a uniform grid across the image
#2. also sample some points as in strategy 1, but select additional points around the pen

#See fucntions in pen_functions.py

points1 = sample_points_grid(I, WIN_SIZE) # sampling strategy 1
points2 = sample_points_around_pen(I, p1, p2, WIN_SIZE) # sampling strategy 2
if display_images:
    # plot both sampling strategies in a single figure using subplots
    plt.figure(figsize=(10,12))
    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.plot(points1[:,0], points1[:,1], 'w.')

    plt.subplot(1,2,2)
    plt.imshow(I)
    plt.plot(points2[:,0], points2[:,1], 'w.')

# Additionally the patches need to be labeled 
# hence in order to do this the distance of the patch's center to the tip of the pen (class 1),
# to the end of the pen (class 2), or to the middle of the pen (class 3) is considered
# See 

CLASS_NAMES = [
    'background', # class 0
    'tip',        # class 1
    'end',        # class 2
    'middle'      # class 3
]

def plot_labeled_points(points, labels):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'r.', label=CLASS_NAMES[0])
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'g.', label=CLASS_NAMES[1])
    plt.plot(points[labels == 2, 0], points[labels == 2, 1], 'b.', label=CLASS_NAMES[2])
    plt.plot(points[labels == 3, 0], points[labels == 3, 1], 'y.', label=CLASS_NAMES[3])

labels1 = make_labels_for_points(I, p1, p2, points1, WIN_SIZE)
labels2 = make_labels_for_points(I, p1, p2, points2, WIN_SIZE)
if display_images:    
    plt.figure(figsize=(10,12))
    
    plt.subplot(1,2,1)
    plt.imshow(I)
    plot_labeled_points(points1, labels1)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.imshow(I)
    plot_labeled_points(points2, labels2)
    plt.legend()
    
class_counts1 = count_classes(labels1)
class_counts2 = count_classes(labels2)
print('class occurrences with strategy 1:', class_counts1)
print('class occurrences with strategy 2:', class_counts2)

#As it can be seen using strategy 1 one classes is very frequent, and the others much less so. With strategy 2, the classes are more uniformly distributed.

#The amount of 'uniformity' in a distribution is measured using entropy which measures the amount of suprise or uncertainty we would have about the outcome if we would sample from a given distribution.

ANSWER_STRATEGY1_ENTROPY = entropy(class_probs(class_counts1))
ANSWER_STRATEGY2_ENTROPY = entropy(class_probs(class_counts2))
ANSWER_MAX_FOUR_CLASS_ENTROPY = entropy(np.array([0.25, 0.25, 0.25, 0.25]))

print('Entropy for labels in strategy 1:', ANSWER_STRATEGY1_ENTROPY)
print('Entropy for labels in strategy 2:', ANSWER_STRATEGY2_ENTROPY)
print('max. Entropy for four classes distribution:', ANSWER_MAX_FOUR_CLASS_ENTROPY)

# Perform all steps together
#See functions extract_patches and extract_multiple_images

"""Prepare Training and Testing Data"""

# Split 26 training images and 11 testing images
train_imgs = list(range(0,26))
test_imgs = list(range(26,len(Is)))

X_train, y_train, points_train, imgids_train = extract_multiple_images(train_imgs, Is, annots, WIN_SIZE, FEAT_SIZE)

X_test, y_test, points_test, imgids_test = extract_multiple_images(test_imgs, Is, annots, WIN_SIZE, FEAT_SIZE)

"""Train and Evaluate 3 different models"""

# sklearn imports
from sklearn.linear_model import SGDClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sgd_clf = SGDClassifier(random_state=42, loss='log') # this should be the Logistic Regression Classifier
dt_clf = DecisionTreeClassifier(random_state=42) # this should be the Decision Tree classifer
rf_clf = RandomForestClassifier(random_state=42) # this should be the Random Forest classifier

sgd_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

#Save models
if save_model==True:
    pickle.dump(sgd_clf, open(f'saved_models/sgd.sav', 'wb'))
    pickle.dump(dt_clf, open(f'saved_models/dt.sav', 'wb'))
    pickle.dump(rf_clf, open(f'saved_models/rf.sav', 'wb'))


def eval_classifier(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    confmat = confusion_matrix(y, y_pred)
    
    return accuracy, confmat

def report_eval(name, accuracy, confmat):
    print(f'*** {name} ***')
    print(f' confusion matrix:')
    print(confmat)
    print(f' accuracy: {accuracy:.3f}')
    print()


# Performance on Training Data
print('-- TRAINING data evaluation --')
print()

# logistic regression
sgd_train_accuracy, sgd_train_confmat = eval_classifier(sgd_clf, X_train, y_train)
report_eval('Logistic Regression', sgd_train_accuracy, sgd_train_confmat)

# decision tree
dt_train_accuracy, dt_train_confmat = eval_classifier(dt_clf, X_train, y_train)
report_eval('Decision Tree', dt_train_accuracy, dt_train_confmat)

# random forest
rf_train_accuracy, rf_train_confmat = eval_classifier(rf_clf, X_train, y_train)
report_eval('Random Forest', rf_train_accuracy, rf_train_confmat)

# Performance on Test Data
print('-- TEST data evaluation --')
print()

# logistic regression
sgd_test_accuracy, sgd_test_confmat = eval_classifier(sgd_clf, X_test, y_test)
report_eval('Logistic Regression', sgd_test_accuracy, sgd_test_confmat)

# decision tree
dt_test_accuracy, dt_test_confmat = eval_classifier(dt_clf, X_test, y_test)
report_eval('Decision Tree', dt_test_accuracy, dt_test_confmat)

# random forest
rf_test_accuracy, rf_test_confmat = eval_classifier(rf_clf, X_test, y_test)
report_eval('Random Forest', rf_test_accuracy, rf_test_confmat)

"""Visualize Results"""

def plot_image_classification_results(clf, img_idx, Ps_test, labels_test, points_test, imgids_test):
    mask = imgids_test == img_idx

    y_test_pred = clf.predict(Ps_test[mask])
    y_test_pred_prob = clf.predict_proba(Ps_test[mask])
    points = points_test[mask,:]

    confmat = confusion_matrix(labels_test[mask], y_test_pred)
    accuracy = accuracy_score(labels_test[mask], y_test_pred)

    print(f' confusion matrix:')
    print(confmat)
    print(f' accuracy: {accuracy:.3f}')

    best_idx1 = y_test_pred_prob[:,1].argmax()
    best_idx2 = y_test_pred_prob[:,2].argmax()
    
    # load image
    I = Is[img_idx]

    plt.figure()
    plt.imshow(I)
    plt.plot(points[y_test_pred==0, 0], points[y_test_pred==0, 1], '.r')
    plt.plot(points[y_test_pred==3, 0], points[y_test_pred==3, 1], '.y')
    plt.plot(points[y_test_pred==1, 0], points[y_test_pred==1, 1], '.g')
    plt.plot(points[y_test_pred==2, 0], points[y_test_pred==2, 1], '.b')
    plt.plot(points[(best_idx1, best_idx2), 0], points[(best_idx1, best_idx2), 1], 'c-', linewidth=2)
    plt.plot(points[best_idx1, 0], points[best_idx1, 1], 'co')
    plt.show()

test_img_idxs = np.unique(imgids_test)
classifiers = {'Logistic Regression': sgd_clf, 'Random Forest': rf_clf, 'Decision-Tree': dt_clf}

def plot_nth_test_result(clf, n):
    plot_image_classification_results(clf, test_img_idxs[n], X_test, y_test, points_test, imgids_test)


print("Which classifier? (value between 1 and 3)")
print("1. Logistic Regression")
print("2. Random Forest")
print("3. Decision-Tree")    

while True:
    clf = int(input())
    if clf==1:
        clf = sgd_clf
        break
    elif clf==2:
        clf = rf_clf
        break
    elif clf==3:
        clf = dt_clf
        break
    else:
        print("Error enter a value between 1 and 3")

print("Choose which test image between 1 and 11")

while True:
    n = int(input())
    if n not in list(range(1,12)):
        print("Error enter a value betwen 1 and 11")
    else:
        break

n = n-1

plot_nth_test_result(clf, n)














