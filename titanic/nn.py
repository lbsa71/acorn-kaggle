import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(15324)

def normalizeColumn(dataSet, column):
    dataSet[column] = dataSet[column] / np.max(dataSet['age_processed'])
    return dataSet

def getProcessedData(dataSet):
    normalizeColumn(dataSet, 'age_processed')
    dataSet['class_category'] = dataSet['Pclass'].astype('category')
    return pd.get_dummies(dataSet.loc[:, ['age_processed', 'class_category', 'Sex', 'Parch', 'SibSp', 'Embarked']]).astype('float32')

def castObjectToCategory(data):
    for i in range(len(data.columns)):
        if data.dtypes[i] == 'object':
            name = data.columns[i]
            data.iloc[:,i] = data.iloc[:,i].astype('category')

data = pd.read_csv('./data/train.csv')

data = data.dropna(subset=['Pclass', 'Age', 'Sex', 'Parch']).reset_index()

# females = data[data['Sex'] == 'female'].reset_index()
# allMales = data[data['Sex'] == 'male'].reset_index()
# males = allMales.truncate(after=len(females))
# data = females.append(males, ignore_index = True)
# data = data.sample(frac = 1).reset_index(drop = True)
print(data.head())
print('Items: ' + str(len(data)))

castObjectToCategory(data)

print(data.dtypes)
print(data.head(5))

data['y'] = data['Survived'].astype('category')
data['age_processed'] = data['Age']
processed = getProcessedData(data)
print(processed.head())
print(processed.dtypes)

train_idx = pd.Series(np.random.rand(len(data))) > 0.7

y_processed = pd.get_dummies(data.loc[:,['y']])

x_train = processed[train_idx].as_matrix().astype('float32')
y_train = y_processed[train_idx].as_matrix().astype('float32')
x_test = processed[~train_idx].as_matrix().astype('float32')
y_test = data[~train_idx].as_matrix(['y']).astype('float32')

x_shape = len(processed.columns)

def model_fn(features, labels, mode):
    hidden_nodes = 10
    hidden_nodes2 = 5
    W_h = tf.Variable(np.random.randn(x_shape, hidden_nodes), dtype=tf.float32) 
    b_h = tf.Variable(np.zeros(hidden_nodes), dtype=tf.float32)
    W_h2 = tf.Variable(np.random.randn(hidden_nodes, hidden_nodes2), dtype=tf.float32) 
    b_h2 = tf.Variable(np.zeros(hidden_nodes2), dtype=tf.float32)
    W_o = tf.Variable(np.random.randn(hidden_nodes2, 2), dtype=tf.float32)
    b_o = tf.Variable(np.zeros(2), dtype=tf.float32)

    y_h2 = tf.sigmoid(tf.matmul(features['x'], W_h) + b_h)
    y_h = tf.sigmoid(tf.matmul(y_h2, W_h2) + b_h2)
    y_raw = tf.matmul(y_h, W_o) + b_o

    y_ = tf.nn.softmax(y_raw)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = y_)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_raw, labels=labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = y_,
        loss = loss,
        train_op = train)

input_train_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
    batch_size=4, num_epochs=None, shuffle=True)
input_test_fn = tf.estimator.inputs.numpy_input_fn({'x': x_test},
    batch_size=1, num_epochs=1, shuffle=False)

estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir='./models')
estimator.train(input_fn = input_train_fn, steps=10000)

test_labels = np.argmax(list(estimator.predict(input_fn = input_test_fn)), 1)
y_test = np.transpose(y_test)[0]

print(test_labels[0:15])
print(y_test[0:15].astype('int32'))
print(np.mean(np.abs(test_labels - y_test)))

# train_metrics = estimator.evaluate(input_fn = input_train_fn)
# print('Eval metrics: %r'%train_metrics)
