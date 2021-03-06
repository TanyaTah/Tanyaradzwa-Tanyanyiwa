#!/usr/bin/env python
# coding: utf-8

#Load data
from sklearn import datasets
iris = datasets.load_iris()
#or
#from sklearn.datasets import load_iris
#iris = load_iris

#Determine the classifier algorithm
from sklearn.neighbors import KNeighborsClassifier
#or
#from sklearn import neighbors

#A complete breakdown of the data within the notebook
iris.keys()

#Target names
iris['target_names']

#Feature names
iris['feature_names']

#Type
type(iris['data'])

#Shape
iris['data'].shape

#Finding number of neighbors
import numpy as np
neighbors = np.arange(1,9) 
#Accuracy
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

#Visualizing Accuracy and number of neighbors
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#Create the model object using the specific algorithm, that is, KNeighbors
for i,k in enumerate(neighbors): knn_model = KNeighborsClassifier(n_neighbors=k)
print(knn_model)

#Split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=0)

#Fit the Model on the training data
knn_model.fit(X_train, y_train)

#Use the model to predict the labels of the test data
y_predicted = knn_model.predict(X_test)

#View predicted data
y_predicted

# Check the results using Classification Report
from sklearn import metrics
y_predicted =knn_model.predict(X_test)
print(metrics.classification_report(y_test, y_predicted))
#The averaged f1-score is often used as a convenient measure of the overall performance of an algorithm,that is , macro avg

#Confusion matrix
print(metrics.confusion_matrix(y_test, y_predicted))

#Confusion matrix as a crosstab
pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)

#View probabilities
y_predicted_probability = knn_model.predict_proba(X_test)[:,1]
print(y_predicted_probability)

#Other performance scores
import numpy as np

print("Test set score: {:.2f}".format(np.mean(y_predicted == y_test)))

print("Test set score: {:.5f}".format(knn_model.score(X_test, y_test)))

