# Spam Email Classification with Scikit-Learn

For the implementation of the algorithms in this exercise, functions from the Scikit-Learn library will be used.

## 1. Spam Email Filtering
In this exercise, the Naive Bayes algorithm and Support Vector Machines (SVM) will be used to build a classifier-filter for unwanted electronic messages. 

### Data Preparation
- Each message's words are separated by one or more spaces. 
- Each message will be represented differently from what was described in the lesson. A dictionary (V) will be created with all the words from all the messages in the training set. 
- A message with d words is represented by a vector (x1, x2, ..., xd) of length d. 
- Words appearing in at least 5 messages will be recorded in the dictionary. Additionally, all uppercase letters will be converted to lowercase (normalization).

## 2. Naive Bayes Algorithm
From Scikit-Learn, use the function implementing the Naive Bayes classifier with multinomial distribution and Laplace smoothing. Use the spam_train.txt file for model training and spam_test.txt for evaluating its efficiency and calculating the error.

### Intuition
Certain keywords are highly indicative of which class a message belongs to. One way to assess how indicative word i is by computing:
log( p(xj=i | y=1) / p(xj=i | y=0) )
Use the above formula to calculate the 5 most indicative words.

## 3. Support Vector Machines (SVM)
Utilize the functions provided by Scikit-Learn for SVM classifiers to classify messages as normal or spam. Use a Gaussian kernel (Radial Basis Function) and experiment with the kernel's parameters. Find the kernel radius that yields the smallest error with respect to the test set.

