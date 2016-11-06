from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import tensorflow.contrib.layers.python.layers as layers
import tensorflow.contrib.learn.python.learn as learn
from  tensorflow.python.ops import parsing_ops
import numpy as np
from tensorflow.python.framework.ops import Tensor

def input_fn(input_data, output_data):
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(len(input_data[k]))],
        values=input_data[k],
        shape=[len(input_data[k]), 1])
                      for k in [0,1,2]}
    # Merges the two dictionaries into one.
    feature_cols = dict(categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(output_data)
    # Returns the feature columns and the label.
    return feature_cols, label


def my_model(features, labels):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Convert the labels to a one-hot tensor of shape (length of features, 3) and
    # with a on-value of 1 for each one-hot vector of length 3.
    #   if(labels != None):
    #       labels = tf.one_hot(labels, 3, 1, 0)
    #   else:
    #       parsing_ops.FixedLenFeature(shape= shape[1:], dtype=int32)
    labels = tf.one_hot(labels, 3, 1, 0)
    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    features = layers.stack(features, layers.fully_connected, [10, 20, 10])

    # Create two tensors respectively for prediction and loss.
    prediction, loss = (
        tf.contrib.learn.models.logistic_regression(features, labels)
    )
    # Create a tensor for training op.
    train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)

    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

def train():
    iris = datasets.load_iris()
    classifier = learn.Estimator(model_fn=my_model,
                                 model_dir="/home/dev/TensorLearningCodes/chap19_model/")
    classifier.fit(iris.data, iris.target, steps=1000)
    y_predicted = [
      p['class'] for p in classifier.predict(iris.data, as_iterable=True)]
    score = metrics.accuracy_score(iris.target, y_predicted)
    print('Accuracy: {0:f}'.format(score))

def predict():
    iris = datasets.load_iris()
    classifier = learn.Estimator(model_fn=my_model,
                                 model_dir="/home/dev/TensorLearningCodes/chap19_model/")
    #classifier.fit(iris.data, iris.target, steps=1000)
    y_predicted = [
      p['class'] for p in classifier.predict(iris.data, as_iterable=True)]
    score = metrics.accuracy_score(iris.target, y_predicted)
    print('Accuracy: {0:f}'.format(score))


predict()