---
layout: post
title: Building a Linear Gaussian Classifier to Classify Cat and Grass
author: Alex Xu
date: '2019-11-04 14:35:23'
category: classifier
summary: Develop a linear classifier using Gaussian model to classify image of cat and grass.
thumbnail: catgrass.png
---

This blog shows the steps needed to build a Gaussian classifier using pre-computed data to classify whether a pixel in an image belongs to a cat or grass. The goal of the project is to show a use case of the maximum-a-posteriori model of a simple classification task. This does not involve any neural network or deep learning. The entire project is implemented using python.

This project and all associated data are provided by Prof. Stanley Chan from Purdue University for learning purposes only.

### **Part 0** - Table of Contents
[1 - Setting Up](#setup)<br>
[2 - Theory](#theory)<br>
[3 - Building the Classifier](#classification)<br>
[3.0 - Building the Classifier](#30)<br>
[3.1 - Building the Classifier](#31)<br>
[3.2 - Building the Classifier](#32)<br>
[3.3 - Building the Classifier](#33)<br>
[3.3.1 - Building the Classifier](#331)<br>
[3.3.2 - Building the Classifier](#332)<br>
[3.3.3 - Building the Classifier](#333)<br>
[3.3.4 - Building the Classifier](#334)<br>
[4 - Evaluation the Classifier](#evaluation)<br>
[5 - Discussion](#discussion)<br>
[6 - Appendix](#appendix)<br>

<a name='setup'>

### **Part 1** - Setting Up

This project is done in python. I'm using python version 3.7 but other versions of python should also work. In the case of python 2.x, there might be some differences in syntax. Python installation instruction can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download).

The following two packages are also used in this project: __numpy__ and __matplotlib__. Those can be installed by running:

`pip install numpy`
`pip install matplotlib`

The entire data packages, including the images and raw data needed for building the classifier, can be downloaded [here](/assets/data/catgrass_data.zip). The zip file contains:
1. cat_grass.jpg: the input image to be classified;
2. truth.png: the ground truth of the classification;
3. train_cat.txt: the training data for cat model;
4. train_grass.txt: the training data for grass model.

The use of those four files will be explained later.

<a name='theory'>

### **Part 2** - Theory

Both __train_cat.txt__ and __train_grass.txt__ can be interpreted as raw cvs files with 64 rows and n columns. Each column represents the pixel value for an 8x8 patch in the image that corresponds to each class. The theory here is that those patches from each class (cat or grass) behave like a Gaussian model with some mean and standard deviation:

<img src="https://latex.codecogs.com/gif.latex?$$\pmb{\mu}_{(\text{class})}=\frac{1}{K}\sum_{k=1}^{K}\pmb{x}_{k}^{(\text{class})}$$" /> <br>
and <br>
<img src="https://latex.codecogs.com/gif.latex?$$\pmb{\Sigma}_{(\text{class})}=\frac{1}{K}\sum_{k=1}^{K}\left(\pmb{x}_{k}^{(\text{class})}-\pmb{\mu}_{(\text{class})}\right)^T\left(\pmb{x}_{k}^{(\text{class})}-\pmb{\mu}_{\text{class}}\right)$$" /> 

where K is the total number of training examples (number of numbers per row in the given file). Based on this model, assume that an unknown patch is from certain class, the probability of seeing that patch __z__ can be modeled as conditional probability:

<img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{class i}\right)=\frac{1}{(2\pi)^{d/2}\left|\pmb{\Sigma}_{\text{class i}}\right|^{1/2}}\exp\left\{-\frac{1}{2}\left(\pmb{z}-\pmb{\mu}_{\text{class i}}\right)^T\pmb{\Sigma}_{\text{class i}}^{-1}\left(\pmb{z}-\pmb{\mu}_{\text{class i}}\right)\right\}" />

By the [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem), the probability of a patch __z__ falls under certain class is:

<img src="https://latex.codecogs.com/gif.latex?f_{\text{class}|\pmb{Z}}\left(\text{class i}|\pmb{z}\right)=\frac{f_{\pmb{z}|\text{class}}\left(\pmb{z}|\text{class i}\right)f\left(\text{class}\right)}{f_{\pmb{Z}}(\pmb{z})}" />

where <img src="https://latex.codecogs.com/gif.latex?f_{\text{class}}\left(\text{class}\right)" /> is our prior knowledge about the problem: without looking at what the patch look like, what is the probability that it is from a certain class. In our example, this can be seen as the ratio of total training example that belongs to that class:

<img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\text{class k}\right)=\frac{K^{\text{(class k)}}}{\sum_{\text{classes}} K}" />

In our case will simply be the ratio between cat training samples to the total training samples and grass training samples to the total training samples. For our classifier, there are two possible classes: cat or grass; therefore we are comparing <img src="https://latex.codecogs.com/gif.latex?f_{\text{class}|\pmb{Z}}\left(\text{cat}|\pmb{z}\right)" /> vs. <img src="https://latex.codecogs.com/gif.latex?f_{\text{class}|\pmb{Z}}\left(\text{grass}|\pmb{z}\right)" />. We can make conclusion as following:

<img src="https://latex.codecogs.com/gif.latex?\pmb{z}=\begin{cases}\text{cat}&\text{if\ }f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{cat}\right)f\left(\text{cat}\right)>f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{grass}\right)f\left(\text{grass}\right)\\\text{grass}&\text{if\ }f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{cat}\right)f\left(\text{cat}\right)<f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{grass}\right)f\left(\text{grass}\right)\end{cases}" />

This decision rule based on above equation is called **Maximum-a-Posteriori (MAP)** decision.

<a name='classification'>

### **Part 3** - Building the Classifier 

This is a long precedure. I'm breaking it down into separate steps.

<a name='30'>

#### **Part 3.0** - Import

Before writing any of the actual program, two installed libraries are needed. We will also import the built in module __time__ to have a rough idea of how long does the program takes.
```python
import numpy as np
import matplotlib.pyplot as plt
import time
```

<a name='31'>

#### **Part 3.1** - Load the files

We can load the images using the matplotlib library and the training texts using numpy library. When the jpg image is loaded the default value will be from 0 to 255. We have to rescale it to [0, 1] instead. This does not apply to the png file. For the ground truth file, we have to round each value to 0 or 1 before using it. The image contains values like 0.1 for black pixel and 0.8 for white pixels. All that we care is whether it is white (1) or black (0).
```python
img_raw = plt.imread('data/cat_grass.jpg') / 255.0  # load the raw image and scale it to [0,1]
img_truth = np.round(plt.imread('data/truth.png'))  # load the ground truth image and round them to 0 or 1
train_cat = np.loadtxt('data/train_cat.txt', delimiter=',')
train_grass = np.loadtxt('data/train_grass.txt', delimiter=',')
```

<a name='32'>

#### **Part 3.2** - Compute mean vector and covariance matrix

Both mean and covariance can be computed easily utilizing the numpy library. The command __np.mean__ can take one additional parameter indicate the axis. Axis 1 indicates we are looking for a mean column instead of mean row.
```python
def compute_mean_cov(train_data):
    # given the training data, compute the mean and covariance
    mu = np.mean(train_data, axis=1)
    sigma = np.cov(train_data)
    return len(train_data[0]), mu, sigma

K_cat, mu_cat, sigma_cat = compute_mean_cov(train_cat)
K_grass, mu_grass, sigma_grass = compute_mean_cov(train_grass)
```

<a name='33'>

#### **Part 3.3** - Classification and decision making

This part builds the actual classifier and will be broken down into several parts: optimizations can be performed to reduce the runtime and three different implementations.

<a name='331'>

##### **Part 3.3.1** - Obervations and optimizations
This part will be the most important part for the project. The following is a general precedure for classification:
1. initialize the array to store the outputs
2. precompute few constants to reduce the computation cost
3. for each 8x8 patch
    1. flatten the patch to 64x1
    2. calculate <img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{cat}\right)f\left(\text{cat}\right)" /> and <img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{grass}\right)f\left(\text{grass}\right)" /> and compare
    3. assign the value to the output array: 1 means cat and 0 means grass
4. return the output array

There are few observations/optimization we can do here. 
1. Comparing <img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{cat}\right)f\left(\text{cat}\right)" /> and <img src="https://latex.codecogs.com/gif.latex?f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{grass}\right)f\left(\text{grass}\right)" /> holds the same result as comparing <img src="https://latex.codecogs.com/gif.latex?\log\left(f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{cat}\right)f\left(\text{cat}\right)\right)" /> and <img src="https://latex.codecogs.com/gif.latex?\log\left(f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{grass}\right)f\left(\text{grass}\right)\right)" /> since __log__ is a monotonically increasing function.
2. <img src="https://latex.codecogs.com/gif.latex?\log\left(f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{class i}\right)f\left(\text{class i}\right)\right)=\log f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{class i}\right)+\log f\left(\text{class i}\right)" />.
3. <img src="https://latex.codecogs.com/gif.latex?\log f\left(\text{cat}\right)" /> and <img src="https://latex.codecogs.com/gif.latex?\log f\left(\text{grass}\right)" /> are constants throughout the compuation, so we can precompute both values once and simply use them after.
4. <img src="https://latex.codecogs.com/gif.latex?\log f_{\pmb{Z}|\text{class}}\left(\pmb{z}|\text{class i}\right)=-\log\left((2\pi)^{d/2}\left|\pmb{\Sigma}_{\text{class i}}\right|^{1/2}\right)-\frac{1}{2}\left(\pmb{z}-\pmb{\mu}_{\text{class i}}\right)^T\pmb{\Sigma}_{\text{class i}}^{-1}\left(\pmb{z}-\pmb{\mu}_{\text{class i}}\right)" />  holds true since <img src="https://latex.codecogs.com/gif.latex?\log(ab)=\log a+\log b" /> and <img src="https://latex.codecogs.com/gif.latex?\log(\exp(x))=x" />.
5. The entire first term in previous equation <img src="https://latex.codecogs.com/gif.latex?-\log\left((2\pi)^{d/2}\left|\pmb{\Sigma}_{\text{class i}}\right|^{1/2}\right)" /> is an constant; therefore can be precomputed for once for the cat class and once for the grass class to avoid expensive calculation inside the loop.
6. <img src="https://latex.codecogs.com/gif.latex?\pmb{\Sigma}_{\text{class i}}^{-1}" /> is also a constant and can be precomputed before the loop for both classes.

<a name='332'>

##### **Part 3.3.2** - Non-overlapping classifier
Since we are classifying each 8x8 patch, there are different ways of implementing the classifier. One of them will be analyze all non-overlapping patches and classify all 64 pixels into one class for each patch (ignore the edges when the side is not divisible by 8). The shape of the input image for our example is (375, 500) in terms of number of rows and number of columns, and the shape of the output classification matrix will be (368, 400) by ignoring the sides. The following code shows the implementaiton: 

```python
def classifier_nonoverlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, pi_cat, pi_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M // 8 * 8, N // 8 * 8))   # initialize the classification result matrix
    for i in range(M // 8):
        for j in range(N // 8):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability and assign to all 64 pixels
            if (fcat_z > fgrass_z):
                result[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = 1

    return result
```

<a name='333'>

##### **Part 3.3.3** - Overlapping classifier
Another way will be classify each pixel based on one 8x8 patch and patches can overlap with one another. The resulting matrix will be size (368, 493). The following code shows the implementaiton: 
```python
def classifier_overlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M - 8 + 1, N - 8 + 1))   # initialize the classification result matrix

    for i in range(M - 8 + 1):
        for j in range(N - 8 + 1):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i : i + 8, j : j + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability
            if (fcat_z > fgrass_z):
                result[i][j] = 1

    return result
```

<a name='334'>

##### **Part 3.3.4** - Improved overlapping classifier

This is combining both methods together for classification. The classifier will have overlapping 8x8 patches when looping through the input image but it will add the result of the patch to all 64 pixels. Therefore, most of the pixels will be classified 64 times with different patch while pixels on the edge will be classified less than that. At the end, each pixel will have a majority vote on which class it belongs to based on all classifications performed. The resulting image will have the same shape as the input image (375, 500).
```python
def classify_improved_overlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M, N))   # initialize the classification result matrix
    count = np.zeros((M, N))

    for i in range(M - 8 + 1):
        for j in range(N - 8 + 1):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i : i + 8, j : j + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability
            count[i : i + 8, j : j + 8] += 1
            if (fcat_z > fgrass_z):
                result[i : i + 8, j : j + 8] += 1

    return np.round(result / count)
```
<a name='evaluation'>

### **Part 4** - Evaluating the Classifier
Given the classifier, we can evaluate the runtime and accuracy of the classifier. Following piece fo code is used for evaluation. The metric used here for accuracy is simply the number of correctly classified pixels to total number of pixels. When evaluating the non-overlapping implementation, we are ignoring the right and bottom edges of the picture. When evaluating the overlapping implementation, we are ignoring first 4 rows and columns, and last 3 rows and columns; this is chosen to incorperate the idea that we are classifying each center pixel by the 8x8 patch arounds it. Switching the 3 and 4 around or using a different way of comparing the result with the ground truth is totally fine.
```python
def eval(classifier, img_raw, img_truth, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, save_img=''):
    '''
    This method evaluates the classifier and calculate the error with the training model. The last argument is optional to save the result image.
    The first argument classifier is the classification function
    '''
    start_time = time.time()
    result = classifier(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass)
    end_time = time.time()
    accuracy = np.sum(np.sum(result == img_truth) / (len(result) * len(result[0])))
    
    if save_img != '':      # if the save file is given
        plt.imsave(save_img, result * 255, cmap='gray')

    return end_time - start_time, accuracy

M = len(img_raw)
N = len(img_raw[0])

runtime, accuracy = eval(classifier_nonoverlap, img_raw, img_truth[:M // 8 * 8, :N // 8 * 8], mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'non-overlapping.png')
print(f'Non-overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')

runtime, accuracy = eval(classifier_overlap, img_raw, img_truth[4:-3, 4:-3], mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'overlapping.png')
print(f'Overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')

runtime, accuracy = eval(classify_improved_overlap, img_raw, img_truth, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'overlapping_improved.png')
print(f'Improved overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')
```

Following are my outputs:
```
Non-overalapping classifier takes 0.054 seconds to run and the accuracy is 0.906
Overalapping classifier takes 1.400 seconds to run and the accuracy is 0.913
Improved overalapping classifier takes 2.192 seconds to run and the accuracy is 0.923
```

and those are my images:

<div style="align: left; text-align:center;">
  <img src="/assets/img/posts/catgrass/non-overlapping.png">
  <figcaption>Fig.1 - Output image from the non-overlapping classifier.</figcaption>
</div>
<br>
<div style="align: left; text-align:center;">
  <img src="/assets/img/posts/catgrass/overlapping.png">
  <figcaption>Fig.2 - Output image from the overlapping classifier.</figcaption>
</div>
<br>
<div style="align: left; text-align:center;">
  <img src="/assets/img/posts/catgrass/overlapping_improved.png">
  <figcaption>Fig.3 - Output image from the improved overlapping classifier.</figcaption>
</div>
<br>

<a name='discussion'>

### **Part 5** - Discussion
The classifier performs well on the test image, but if you try with other images, it probably fails as shown in Figure 4. Why?
<div style="align: left; text-align:center;">
  <img src="/assets/img/posts/catgrass/outsample.png" width="100%">
  <figcaption>Fig.4 - Running the classifier on the image shown on the left and the result is shown on the right.</figcaption>
</div>
<br>

Couple possible reasons are:
1. The training data are generated based on cat_grass.jpg this single image. Using such biased training set and do well on the training data does not guarantee good performance on other images.
2. The entire classifier is built on the assumption that the distribution of the texture of cat and grass are gaussian. In reality this is unlikely, expecially for things like cat where there are so many different fur patterns.

<a name='appendix'>

### **Appendix** - Source Code
```python
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_mean_cov(train_data):
    # given the training data, compute the mean and covariance
    mu = np.mean(train_data, axis=1)
    sigma = np.cov(train_data)
    return len(train_data[0]), mu, sigma

def classifier_nonoverlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, pi_cat, pi_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M // 8 * 8, N // 8 * 8))   # initialize the classification result matrix
    for i in range(M // 8):
        for j in range(N // 8):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability and assign to all 64 pixels
            if (fcat_z > fgrass_z):
                result[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = 1

    return result

def classifier_overlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M - 8 + 1, N - 8 + 1))   # initialize the classification result matrix

    for i in range(M - 8 + 1):
        for j in range(N - 8 + 1):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i : i + 8, j : j + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability
            if (fcat_z > fgrass_z):
                result[i][j] = 1
    return result

def classify_improved_overlap(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass):
    # find the dimension of the image
    M = len(img_raw)
    N = len(img_raw[0])

    # calculate log(f(class)) as described in item 3
    fcat = np.log(K_cat / (K_cat + K_grass))
    fgrass = np.log(K_grass / (K_cat + K_grass))

    # calculate the constent as described in item 4
    coef_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_cat))))
    coef_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(sigma_grass))))

    # calculate the inverse of the covariance matrix as described in item 6
    inv_cat = np.linalg.pinv(sigma_cat)
    inv_grass = np.linalg.pinv(sigma_grass)

    result = np.zeros((M, N))   # initialize the classification result matrix
    count = np.zeros((M, N))

    for i in range(M - 8 + 1):
        for j in range(N - 8 + 1):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = img_raw[i : i + 8, j : j + 8]
            z = z.flatten('F')
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass

            # calculate f_z|class(z|cat) and f_z|class(z|grass)
            fcat_z = fcat + coef_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            fgrass_z = fgrass + coef_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)

            # find out which class has higher probability
            count[i : i + 8, j : j + 8] += 1
            if (fcat_z > fgrass_z):
                result[i : i + 8, j : j + 8] += 1

    return np.round(result / count)

def eval(classifier, img_raw, img_truth, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, save_img=''):
    '''
    This method evaluates the classifier and calculate the error with the training model. The last argument is optional to save the result image.
    The first argument classifier is the classification function
    '''
    start_time = time.time()
    result = classifier(img_raw, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass)
    end_time = time.time()
    accuracy = np.sum(np.sum(result == img_truth) / (len(result) * len(result[0])))
    
    if save_img != '':      # if the save file is given
        plt.imsave(save_img, result * 255, cmap='gray')

    return end_time - start_time, accuracy


if __name__ == '__main__':
    img_raw = plt.imread('cat_grass.jpg') / 255.0  # load the raw image and scale it to [0,1]
    img_truth = np.round(plt.imread('truth.png'))  # load the ground truth image and round them to 0 or 1
    train_cat = np.loadtxt('train_cat.txt', delimiter=',')
    train_grass = np.loadtxt('train_grass.txt', delimiter=',')

    K_cat, mu_cat, sigma_cat = compute_mean_cov(train_cat)
    K_grass, mu_grass, sigma_grass = compute_mean_cov(train_grass)

    M = len(img_raw)
    N = len(img_raw[0])

    runtime, accuracy = eval(classifier_nonoverlap, img_raw, img_truth[:M // 8 * 8, :N // 8 * 8], mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'non-overlapping.png')
    print(f'Non-overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')

    runtime, accuracy = eval(classifier_overlap, img_raw, img_truth[4:-3, 4:-3], mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'overlapping.png')
    print(f'Overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')

    runtime, accuracy = eval(classify_improved_overlap, img_raw, img_truth, mu_cat, mu_grass, sigma_cat, sigma_grass, K_cat, K_grass, 'overlapping_improved.png')
    print(f'Improved overalapping classifier takes {runtime:.3f} seconds to run and the accuracy is {accuracy:.3f}')
```
