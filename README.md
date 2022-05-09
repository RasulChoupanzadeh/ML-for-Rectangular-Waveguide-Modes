# Machine Learning for Detection of Modal Configurations in Rectangular Waveguides
### Author: Rasul Choupanzadeh
### Date: 5/09/2022

This code was written for a class project in the course entitled ECE 504: "Machine Learning for Electromagnetics" that was taught 
in Spring 2022 by Prof. Zadehgol at the University of Idaho in Moscow, Idaho.

# Overview
- There is a code for generating dataset, for training of Machine Learning algorithms, based on the the field equations of rectangular waveguides in [1]. 

- It also implements the Machine Learning algorithms such as Stochastic Gradient Descent (SGD) and K-Nearest Nrighbors (KNN) from [2]. This code is provided based on the Jupyter notebooks in [2-3]. 

- Final Report.pdf is an 8-page report including methodoloy, formulation, and numerical results of proposed Machine Learning models.

- This code has been developed under some important assumptions:
    * It is only trained on the dataset from Transverse Electric (TE) mode configuration in rectangular waveguides, thus, it only predicts TE mode configurations in rectangular waveguides, however, it may be easily developed for Transverse Magnetic (TM) configurations in rectangular or spherical waveguides by changing the equation used in data_generation.py to the desired equations described in [1].
    * It is only trained on the magnitude and phase plots of Electric field in x-direction (Ex), thus, it detects the modal configuration based on Ex dataset, however, it can be expanded to other field datasets such as Ey, Ez, Hx, Hy, Hz by using their field equations in data_generation.py.
    * The generated dataset for Final report.pdf is provided under the assumptions of m,n= 0, 1, 2. It is easily expandable by changing the maximum values of m and n to desired values.
    * The distribution of added noises are cosidered as Exponential and Gaussian distributions.  
    * The codes are wrtitten using scikit-learn library.


# Licensing
From [3], In addition to licensing GNU GPL v.3:

>- If the code is used in a scientific work, then reference should be made as follows:
>     * ML_algorithm.py: References [2], [3].
>     * data_generation.py: References [1].
>- In addition to the codes, if the Final report.pdf is used in a scentific work, then this repository should be referenced directly.

# Files:

## Main Program Files:
- data_generation.py: generates the input dataset for training Machine Learning models [1]
- ML_algorithm.py: trains and evaluates the Machine Learning models such as Stochastic Gradient Descent (SGD) and K-Nearest Neighbors (KNN). [2-3]

## Supporting Program Files/Folders:
- Final report.pdf: 8-page report includes defining the problem, methodology, formulation, numerical result, discussion, and conclusion [1-9].
- Result folder: includes 12 pdf files generated by code.
    * some_mag_plot_clean.pdf: magnitude plot of a random instance in dataset before adding noise.
    * some_phase_plot_clean.pdf: phase plot of a random instance in dataset before adding noise.
    * some_mag_plot_Exponential.pdf: magnitude plot of a random instance in dataset after adding noise with Exponential distribution.
    * some_phase_plot_Exponential.pdf: phase plot of a random instance in dataset after adding noise with Exponential distribution.
    * some_mag_plot_Gaussian.pdf: magnitude plot of a random instance in dataset after adding noise with Gaussian distribution.
    * some_phase_plot_Gaussian.pdf: phase plot of a random instance in dataset after adding noise with Gaussian distribution.
    * Exponential distribution.pdf: histogram of generated noise using Exponential distribution.
    * Gaussian distribution.pdf: histogram of generated noise using Gaussian distribution.
    * confusion_matrix_SGD.pdf: image representation of confusion matrix in SGD model.
    * confusion_matrix_KNeighbors.pdf: image representation of confusion matrix in KNN model.
    * confusion_matrix_errors_SGD.pdf: image representation of error rates for confusion matrix in SGD model.
    * confusion_matrix_errors_KNeighbors.pdf: image representation of error rates for confusion matrix in KNN model.

    
# Run Machine Learning Algorithm
Run the ML_algorithm.py with desired "incr" values. This code calls the data_generation.py for generating datasets used in Final report.pdf, then, trains, evaluates, and tests the models. To make changes in the dataset, parameters in data_generation.py must be changed.

## Inputs:
- a: An scalar value for width of rectangular waveguide
- b: An scalar value for height of recangular waveguide
- m_max: An scalar value for maximum number of 'm' in modal configuration
- n_max: An scalar value for maximum number of 'n' in modal configuration
- incr: An scalar value for number of incremental points. It is the number of points which data is generated along wideth (x-axis) and height (y-axis) of waveguide
- freq: Frequency data in a vector. Operating frequency of waveguide
    * Dimension: number_of_frequency_points


## Outputs:
- Predicted class: An scalar value representing the predicted class of a random instance in dataset using selected model

- Decision Tree scores: A vector of socres of each class for a random instance in dataset using selected model
    * Dimensions: 1 X number of classes
- Confusion matrix: An array confusion matrix of selected model
    * Dimensions: number of classes X number of classes
- Precission score: An scalar value for precision score of selected model
- Recall score: An scalar value for recall score of selected model
- F1 score: An scalar value for F1 score of selected model
- Accuracy: A vector of k-fold cross validation scores 
    * Dimensions: 1 X k


# Python Version Information:
Python 3.7.6

Libraries used:
Scikit-Learn

# References:
```
[1] David. M. Pozar. Microwave Engineering, 4th edition. John Wiley and Sons, 2011.

[2] A. Geron. Hands on Machine Learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems, 2nd edition. O’Reilly, Sebastopol, California, 2019.

[3] ageron, GitHub. Accessed on: May 3, 2022.
    [Online]. Available: https://github.com/ageron/handson-ml2.

[4] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In 2012 Neural Information Processing Systems (NIPS).

[5] G. Hinton, L. Deng, D. Yu, G. Dahl, and et al. Deep neural networks for acoustic modeling in speech recognition. Signal Processing Magazine, 29, 2012.

[6] A. Conneau, H. Schwenk, L. Barrault, and Y. Lecun. Very deep convolutional networks for text classification. arxiv:1606.01781, 2016.

[7] D. Silver, A. Huang, C. G. Maddison, A. Guez, and et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587), 2016.

[8] A. Zadehgol. Reduced-order stochastic electromagnetic macro-models for uncertainty characterization of 3-d band-gap structures, in FDTD. In 2017 IEEE International Symposium on Antennas and Propagation and USNC/URSI National Radio Science Meeting, San Diego, CA, USA.

[9] N. Balakrishnan and Asit P. Basu. Exponential Distribution: Theory, Methods and Applications, 1st edition. CRC Press, 1996.

```
