## Image Classification using Bag of Visual Words and KNN 

This is code to classify images through k-Nearest Neighbour method using their Bag of Visual Words representation. Training and validation data must be in directories segregated by their categories. Clustering is done using sklearn, but kNN is implemented from scratch. Following are the steps to execute - 

Create an instance of the class BovwKnn with the training and validation directory path as parameter to the constructor. Also, pass number of clusters desired and a pretrained model if available -
```python
obj = BovwKnn('./data/train', './data/validation', n_clusters=250)
```

#### Clustering

Perform this step only if pretrained model is not available. This functions produces desired number of clusters on the training data and stores the model in obk.kmeans. Optionally pass proportion parameter betwween 0 and 1 to indicate proportion of training images to be used -
```python
obj.cluster(proportion=0.5)
```
You can save the model using this line of code -
```python
import pickle
pickle.dump(obj.kmeans, open('./kmeans_250.pkl', 'wb'))
```

#### BOVW vector

This function converts any data to BOVW representation. Can pass obj.train_dict and obj.val_dict -
```python
train = obj.convert_to_bovw(obj.train_dict)
test = obj.convert_to_bovw(obj.val_dict)
``` 

#### kNN

Pass the test data to predict the class labels of all validation images using kNN. Can also pass number of neighbours to the k parameter - 
```python
confusion_matrix, prediction_accuracy = obj.knn(train, test, k=10)
``` 
