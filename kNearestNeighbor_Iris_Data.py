#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Load data
from sklearn import datasets
iris = datasets.load_iris()
#or
#from sklearn.datasets import load_iris
#iris = load_iris


# In[ ]:


#Determine the classifier algorithm
from sklearn.neighbors import KNeighborsClassifier
#or
#from sklearn import neighbors


# In[28]:


#A complete breakdown of the data within the notebook
iris.keys()


# In[41]:


#Target names
iris['target_names']
#
#iris.target_names


# In[15]:


#Feature names
iris['feature_names']


# In[29]:


#Type
type(iris['data'])


# In[20]:


#Shape
iris['data'].shape
#or 
#print(iris.data.type) #output is(150, 4), where 150 is n_samples and 4 is n_features 


# In[128]:


import numpy as np
neighbors = np.arange(1,9) 
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[127]:


#Create the model object using the specific algorithm, that is, KNeighbors

for i,k in enumerate(neighbors): knn_model = KNeighborsClassifier(n_neighbors=k)
print(knn_model)


# In[108]:


#Split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=0)


# In[109]:


#Fitting the Model on the training data
knn_model.fit(X_train, y_train)


# In[111]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[119]:


#Use the model to predict the labels of the test data
y_predicted = knn_model.predict(X_test)


# In[120]:


#View predicted data
y_predicted


# In[113]:


# Check the results using metrics
from sklearn import metrics
y_predicted =knn_model.predict(X_test)


# In[114]:


print(metrics.classification_report(y_test, y_predicted))
#The averaged f1-score is often used as a convenient measure of the overall performance of an algorithm,that is , macro avg


# In[115]:


print(metrics.confusion_matrix(y_test, y_predicted))


# In[116]:


#Confusion matrix as a crosstab
pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)


# In[122]:


y_predicted_probability = knn_model.predict_proba(X_test)[:,1]
print(y_predicted_probability)


# In[117]:


import numpy as np
print("Test set score: {:.2f}".format(np.mean(y_predicted == y_test)))


# In[121]:


print("Test set score: {:.5f}".format(knn_model.score(X_test, y_test)))

