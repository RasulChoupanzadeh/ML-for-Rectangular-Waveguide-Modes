""" ML_algorithm.py

This is based on the concepts and codes from [1-2].

Author: Rasul Choupanzadeh
Date: 05/09/2022

[1] A. Geron. Hands on Machine Learning with Scikit-Learn, Keras and 
    TensorFlow: concepts, tools, and techniques to build intelligent 
    systems, 2nd edition. O’Reilly, Sebastopol, California, 2019.

[2] ageron, GitHub. Accessed on: May 3, 2022.
    [Online]. Available: https://github.com/ageron/handson-ml2.

"""




# Python V3.5 is required
import sys
assert sys.version_info >= (3, 5)
import time

IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

import sklearn
assert sklearn.__version__ >= "0.20"

from numpy import unravel_index

# Common imports
import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#-------------------------------------------------------- Data preparation---------------------------------------------------
# The following command will take about 70 minutes, 17 minutes, and 3 minutes for incr=50, incr=25, and incr=10, respectively. Instead of executing the command, we may load the pre-generated data.

incr=50;
exec(open("data_generation.py").read())   
#dataset = np.load('dataset_incr50_0-2.npy')

X_clean = dataset[:, :2*incr*incr]                            # features (clean: without noise)
y_label = dataset[:, 2*incr*incr:2*incr*incr+2]               # labels
y = dataset[:, 2*incr*incr+2:]                                # 8 classes representing 8 lables (i.e., (0,1), (0,2), (1,0), (1,1),(1,2),(2,0),(2,1),(2,2)), respectively.


shuffle_idx = np.random.permutation(len(dataset))
X_clean = X_clean[shuffle_idx]
y_label = y_label[shuffle_idx]
y = y[shuffle_idx]
X_clean.shape
y_label.shape
y.shape


#creating noise
mu, sigma = 0, 1
phase_noise_gauss = np.random.normal(mu, sigma, [X_clean.shape[0],incr*incr])         ## np.random.randn() is a specific normal distribution, with mu=0 and sigma(maximum value of p(x))=1, named standard normal distribution.
mag_noise_gauss = np.random.normal(mu, sigma, [X_clean.shape[0],incr*incr])
noise_gauss = np.hstack((mag_noise_gauss, phase_noise_gauss))


plt.figure()
count, bins, ignored = plt.hist(phase_noise_gauss[0,:], 50, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.xlabel('x')
plt.ylabel('f(x)',rotation=0,labelpad=25)
save_fig("Gaussian distribution")


scale = 1
phase_noise_expo = np.random.exponential(scale, [X_clean.shape[0],incr*incr])         ## np.random.randn() is a specific normal distribution, with location=mu=0 and scale(maximum value of p(x))=1, named standard normal distribution.
mag_noise_expo = np.random.exponential(scale, [X_clean.shape[0],incr*incr])
noise_expo = np.hstack((mag_noise_expo, phase_noise_expo))

plt.figure()
count, bins, ignored = plt.hist(phase_noise_expo[0,:], 50, density=True)
plt.plot(bins, scale * np.exp( - scale * (bins) ), linewidth=2, color='r')
plt.xlim([0,8])
plt.xlabel('x')
plt.ylabel('f(x)',rotation=0, labelpad=25)
save_fig("Exponential distribution")


# adding noise
X = X_clean + noise_expo

import matplotlib as mpl
import matplotlib.pyplot as plt

a = 1.07 * 10**-2
b = 0.43 * 10**-2


some_instance = 3
some_mode = X[some_instance]

## magnitude & phase of clean data
X_mag_clean = X_clean[:,:incr*incr]
X_phase_clean = X_clean[:,incr*incr:]
some_mag_clean = X_mag_clean[some_instance]
some_phase_clean = X_phase_clean[some_instance]
some_mag_image_clean = some_mag_clean.reshape(incr, incr);
some_phase_image_clean = some_phase_clean.reshape(incr, incr);

plt.figure()
plt.imshow(some_mag_image_clean.T, cmap = 'jet', origin = 'lower')
plt.xlabel('x (incr)')
plt.ylabel('y (incr)')
#plt.title('mag_Ex')
save_fig("some_mag_plot_clean")

plt.figure()
plt.imshow(some_phase_image_clean.T, cmap = 'jet', origin = 'lower')
plt.xlabel('x (incr)')
plt.ylabel('y (incr)')
#plt.title('phase_Ex')
save_fig("some_phase_plot_clean")


## magnitude & phase of training dataset (noisy)
X_mag = X[:,:incr*incr]
X_phase = X[:,incr*incr:]
some_mag = X_mag[some_instance]
some_phase = X_phase[some_instance]
print('actual m, n = ', y_label[some_instance])
some_mag_image = some_mag.reshape(incr, incr);
some_phase_image = some_phase.reshape(incr, incr);

plt.figure()
plt.imshow(some_mag_image.T, cmap = 'jet', origin = 'lower')
plt.xlabel('x (incr)')
plt.ylabel('y (incr)')
#plt.title('mag_Ex')
save_fig("some_mag_plot")

plt.figure()
plt.imshow(some_phase_image.T, cmap = 'jet', origin = 'lower')
plt.xlabel('x (incr)')
plt.ylabel('y (incr)')
#plt.title('phase_Ex')
save_fig("some_phase_plot")


# splitting training & test datasets
X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000].ravel(), y[50000:].ravel()


# ----------------------------------------------------SGD Classifier----------------------------------------
start = time.time()

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

sgd_clf.fit(X_train, y_train)                      
predicted_class = sgd_clf.predict([some_mode])

some_mode_scores_sgd = sgd_clf.decision_function([some_mode])
print('some_mode_scores using SGD = ', some_mode_scores_sgd[0])

print('predicted class using SGD : ',predicted_class)

#-------------------------------------------------SGD Performance Measure----------------------------------------------
# Cross validation
from sklearn.model_selection import cross_val_score
cros_vali_sgd = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")     
print('3-fold cross validation scores using SGD classifier: ',cros_vali_sgd)

# Confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix
conf_sgd = confusion_matrix(y_train, y_train_pred_sgd)
print('confusion matrix using SGD classifier: ',conf_sgd)

# Precision
from sklearn.metrics import precision_score, recall_score
prec_sgd = precision_score(y_train, y_train_pred_sgd, average='macro')
print('precision score using SGD classifier: ',prec_sgd)

# Recall
recall_sgd = recall_score(y_train, y_train_pred_sgd, average='macro') 
print('recall score using SGD classifier: ',recall_sgd)

# F1 score
from sklearn.metrics import f1_score
f1_sgd = f1_score(y_train, y_train_pred_sgd, average='macro')
print('f1 score using SGD classifier: ',f1_sgd)

# Error Analysis using confusion matrix image
plt.matshow(conf_sgd, cmap=plt.cm.gray)
save_fig("confusion_matrix_SGD", tight_layout=False)

row_sums = conf_sgd.sum(axis=1, keepdims=True)
norm_conf_sgd = conf_sgd / row_sums

np.fill_diagonal(norm_conf_sgd, 0)
plt.matshow(norm_conf_sgd, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_SGD", tight_layout=False)

end = time.time()
print("The execution time for SGD classifier is :", end-start)


# ------------------------------------------------------KNeighbors classifier--------------------------------------------------
start = time.time()
from sklearn.neighbors import KNeighborsClassifier
y_train_multilabel = np.c_[y_train].ravel()
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train_multilabel)
result_KN = knn_clf.predict([some_mode])
print('predicted class using KNeighbors multilabel classifier:', result_KN[0])
           
#-------------------------------------------------KNeighbors Performance Measure----------------------------------------------
# Cross validation
from sklearn.model_selection import cross_val_score
cros_val_knn = cross_val_score(knn_clf, X_train, y_train, cv=3, scoring="accuracy")     
print('3-fold cross validation scores using KNeighbors classifier: ',cros_val_knn)

# Confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred_knn = cross_val_predict(knn_clf, X_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix
conf_knn = confusion_matrix(y_train, y_train_pred_knn)
print('confusion matrix using KNeighbors classifier: ',conf_knn)

# Precision
from sklearn.metrics import precision_score, recall_score
prec_knn = precision_score(y_train, y_train_pred_knn, average='macro')
print('precision score using KNeighbors classifier: ',prec_knn)

# Recall
recall_knn = recall_score(y_train, y_train_pred_knn, average='macro') 
print('recall score using KNeighbors classifier: ',recall_knn)

# F1 score
from sklearn.metrics import f1_score
f1_knn = f1_score(y_train, y_train_pred_knn, average='macro')
print('f1 score using KNeighors classifier: ',f1_knn)

# Error Analysis using confusion matrix image
plt.matshow(conf_knn, cmap=plt.cm.gray)
save_fig("confusion_matrix_KNeighbors", tight_layout=False)

row_sums = conf_knn.sum(axis=1, keepdims=True)
norm_conf_knn = conf_knn / row_sums

np.fill_diagonal(norm_conf_knn, 0)
plt.matshow(norm_conf_knn, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_KNeighbors", tight_layout=False)

end = time.time()
print("The execution time for KNeighbors multilabel classifier is :", end-start)


#--------------------------------------------------testing the SGD model-----------------------------------------------------------
y_sgd_pred = sgd_clf.predict(X_test)
from sklearn.metrics import accuracy_score
test_acuracy_sgd = accuracy_score(y_test, y_sgd_pred)
print("The accuracy of SGD model for test dataset is :", test_acuracy_sgd)

plt.show()
