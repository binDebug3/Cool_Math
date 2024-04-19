<a name="readme-top"></a>

<div align="center">
    <h1 align="center">Machine Learning Projects</h1>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#recognition">Facial Recognition without Deep Learning</a></li>
    <li><a href="#digits">Classifying Handwritten Digits without ML</a></li>
    <li><a href="#kalman">Learning to Filter Noisy Sensor Data</a></li>
    <li><a href="#linearregression">House Price Prediction with Linear Regression</a></li>
    <li><a href="#logisticregression">Data Classification with Logistic Regression</a></li>
    <li><a href="#pca">Identifying Breast Cancer with PCA and Clustering</a></li>
    <li><a href="#naivebayes">Detecting Spam with Naive Bayes</a></li>
    <li><a href="#randomforest">Determining Species with Random Forests (Decision Trees)</a></li>
    <li><a href="#speech">Speech Recognition like Siri with HMMs</a></li>
  </ol>
</details>


## Project Descriptions

### <a name="recognition">Facial Recognition without Machine Learning</a>
This project uses the Eigenfaces method to recognize faces without using machine learning. The Eigenfaces method involves using the singular value decomposition to find the principal components of a set of images. The principal components are the directions in which the images vary the most. It first computes the mean image, and then subtracts the mean from each image to find the most salient features of each image. This allows the program to find the most similar images overall, while only using linear algebra!

### <a name="digits">Classifying Handwritten Digits with Nearest Neighbors</a>
This project uses the nearest neighbors method to classify handwritten digits without using deep learning. The nearest 
neighbors method involves finding the k nearest neighbors to a point, and then classifying the point based on the class 
of the nearest neighbors. In this project, I used it to classify handwritten digits. I used the MNIST dataset, which is a dataset of 70,000 handwritten digits and found that I could classify the digits with about 90% accuracy.

### <a name="kalman">Learning to Filter Noisy Sensor Data with Kalman Filters</a>
This project uses the Kalman filter to filter noisy sensor data. The Kalman filter is a mathematical method for finding the best estimate of a state based on a series of noisy measurements. In this project, I used it to filter noisy sensor data recording a projectile's trajectory. Using the Kalman filter, I could determine its true position more accurately, extrapolate its point of origin, and predict its future position.

### <a name="linearregression">House Price Prediction with Linear Regression</a>
This project uses linear regression to predict house prices. Linear regression is mathematical method for finding the best
fitting line through a set of data points. In this project, I used it to predict house prices based on the size of the house and
other features.

### <a name="logisticregression">Data Classification with Logistic Regression</a>
This project uses logistic regression to classify data such as flowers into different species. Logistic regression is an 
algorithm that is used to predict the probability of a binary outcome. In this project, I used it to classify flowers into
different species based on their petal and sepal lengths and widths. I used the iris dataset, which is a dataset of 150
flowers with four features each.


### <a name="pca">Identifying Breast Cancer with PCA and Clustering</a>
In this project, I deployed a variety of algorithms from Python's sk_learn package to identify breast cancer. I used
Principal Component Analysis (PCA) to reduce the dimensionality of the data, and then used K-Nearst Neighbors (KNN) and a Random
Forest Classifier to classify the data into benign or malignant tumors. I also used K-Means clustering to cluster the data
into two groups, and then compared the results of the clustering to the actual labels of the data.


### <a name="naivebayes">Detecting Spam with Naive Bayes</a>
This project uses the Naive Bayes algorithm to detect spam emails. The Naive Bayes algorithm is a great way to classify
language data. In this project, I used it to classify emails as spam or not spam based on the words they contain. I used 
the Enron spam dataset, which is a dataset of 33,000 emails that have been labeled as spam or not spam. I found that I could 
classify the emails with about 90% accuracy.


### <a name="randomforest">Determining Animal Species with Random Forests (Decision Trees)</a>
This project uses the Random Forest algorithm to determine the species of animals based on their features. The Random Forest
algorithm machine learning tool that classifies tabular data by creating a large number of decision trees and then averaging 
the results. This algorithm relies on the principle that many weak learners can combine to create a strong learner and works 
well for data that has many features.


### <a name="speech">Speech Recognition like Siri with HMMs</a>
This project uses the Hidden Markov Model (HMM) algorithm to recognize speech. The Hidden Markov Model is a machine learning model
that is used to find hidden variables based on observed time series data. In this case, the observed data is the sound waves of 
speech, and the hidden variables are the words that are being spoken. This is actually the model that Siri first used to recognize 
speech, and it is still used in many speech recognition systems and other applications today.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<hr>

<!-- CONTACT -->
## Contact

Dallin Stewart - dallinpstewart@gmail.com

[![LinkedIn][linkedin-icon]][linkedin-url]
[![GitHub][github-icon]][github-url]
[![Email][email-icon]][email-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[Python-icon]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[NumPy-icon]: https://img.shields.io/badge/NumPy-2596be?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Pandas-icon]: https://img.shields.io/badge/Pandas-120756?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/


[linkedIn-icon]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedIn-url]: https://www.linkedin.com/in/dallinstewart/

[github-icon]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/binDebug3

[Email-icon]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white
[Email-url]: mailto:dallinpstewart@gmail.com
