# Melonama Detection Using Custom Convolutional Neural Network

## Problem statement:

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Project Information

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

Actinic keratosis\
Basal cell carcinoma\
Dermatofibroma\
Melanoma\
Nevus\
Pigmented benign keratosis\
Seborrheic keratosis\
Squamous cell carcinoma\
Vascular lesion

### Loading the data

Data is loaded from the dowloaded dataset. Which is divided into train (2239 Images) and test dataset (118 Images).\
We resized the image to 180 * 180 and divided into training and validation dataset with 20% split with batch size of 32.

### Images after resizing(visulization):

![download](https://github.com/Premparkash/Melonama_CNN/assets/9427473/57f937b8-06d6-48ca-87b3-5366ef8fc514)


### Solution is approached through 3 methods to improve it which is discussed in the document:

#### Output using first baseline method:
![download (1)](https://github.com/Premparkash/Melonama_CNN/assets/9427473/62a3bfbf-dd8c-4480-b5c5-5c1b0ed29dca)



#### Findings:
We can see the case of overfitting here since training accuracy is too high but validation accuracy is too low.
We can use dropout and data augumentation strategy to test if it improves the accuracy of the model
Although training accuracy is 87% but there is alot of chances of wrongly classifying an image due to overfitting.

#### We used ImageDataGenerator and got the output:
![download (2)](https://github.com/Premparkash/Melonama_CNN/assets/9427473/426a16ac-b273-4232-9fda-8c842f2d9b24)


#### Findings:
ImageDataGenerator strategy helped us improve the problem of overfitting.
But accuracy is decresed drastically which can cause misclassification.

#### We used python package known as Augmentor to solve the problem of class imbalance and got the output:
![download (3)](https://github.com/Premparkash/Melonama_CNN/assets/9427473/edf23ce6-5a66-41f0-b2dc-d4c865ce87a8)


#### Findings:
We couldn't completely get rid of overfitting but for sure Class rebalance helped us alot in acchieving better accuracy.
Further techniques like more layers and epochs can be used to improve the model accuracy.
Overall this method is most suitable.


```python

```
