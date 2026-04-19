# cats-dogs-classification using Machine learning
# Overview:
This project presents a binary image classification system that distinguishes between cats and dogs using classical machine learning techniques.

The goal is to classify each image into one of two categories:

* `0` for Cat
* `1` for Dog

A key constraint of this project is that Convolutional Neural Networks (CNNs) are not allowed. For this reason, the solution is based on feature extraction and standard machine learning models.

## Project Objective
The objective of this work is to:
* load and inspect a dataset of cat and dog images
* build a baseline classifier
* improve the performance using better feature extraction
* compare different methods
* define a final prediction function named `cats_dogs_classification(image)`

## Dataset
The dataset is composed of raw images stored in two separate folders:

* `Cat/`
* `Dog/`

The images are not preprocessed and present several challenges:

* different image sizes
* different backgrounds
* varying lighting conditions
* different animal poses and scales
This makes preprocessing and feature extraction essential.

## Methodology

### 1. Baseline approach

The first model was built using raw pixel values.

Each image was:

* resized to `64 x 64`
* flattened into a one-dimensional vector

A Logistic Regression model was then trained on these vectors.

#### Result

The baseline model achieved an accuracy of about `55%`.

This result shows that raw pixels are not sufficiently informative for this task.

---

### 2. Improved approach

To improve performance, a more meaningful image representation was used.

Each image was:

* converted to grayscale
* resized to `64 x 64`
* transformed into HOG features

After that:

* PCA was applied to reduce dimensionality
* a Linear SVM classifier was trained

HOG (Histogram of Oriented Gradients) captures local edge and shape information, which is more useful than raw pixels for distinguishing cats and dogs.

#### PCA:

HOG features can be high-dimensional. PCA reduces feature size while preserving most of the useful information, which helps reduce training time.

#### Linear SVM:

Linear SVM is effective for high-dimensional classification tasks and is computationally more practical than a standard SVM on large datasets.

#### Result

The improved model achieved an accuracy of about `72%`.

This is a significant improvement over the baseline.

---

## Model Comparison

| Model               | Features           | Accuracy |
| ------------------- | ------------------ | -------- |
| Logistic Regression | Raw resized pixels | ~55%     |
| Linear SVM          | HOG + PCA          | ~72%     |

This comparison shows that feature extraction plays a major role in performance.

---

## Final Model

The final selected pipeline is:

1. grayscale conversion
2. resizing to `64 x 64`
3. HOG feature extraction
4. PCA dimensionality reduction
5. Linear SVM classification

This model provided the best balance between accuracy and computation time.

---

## Final Prediction Function

The project includes a function named:

```python
cats_dogs_classification(image)
```

This function:

* takes an image as input
* applies the same preprocessing used during training
* extracts HOG features
* uses the trained model to predict the class
* returns:

  * `0` for Cat
  * `1` for Dog


## Results and Discussion

The baseline model showed that using raw pixels with Logistic Regression is not enough for this classification problem.

The improved pipeline based on HOG, PCA, and Linear SVM performed much better because it uses more meaningful image features.

This project shows that even without deep learning, it is still possible to build a useful image classifier using classical machine learning methods.

---

## Conclusion

This project demonstrates a complete classical machine learning workflow for image classification.

The main conclusions are:

* preprocessing is necessary for image data
* raw pixels are often not enough
* HOG is a strong handcrafted feature extractor for classical computer vision
* PCA helps reduce computation time
* Linear SVM is a good choice for high-dimensional features

