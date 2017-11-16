import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(15324)

# quit()

# print(np.mean(data['Survived']))
# print(np.mean(data['Age']))
# print(np.mean(data['Pclass']))

# undersampling_idx = pd.Series(np.random.rand(len(data))) > 0.7
# data = data[(undersampling_idx) | (data['y'])].reset_index()

# print(np.mean(data['y']))

def normalizeColumn(dataSet, column):
    dataSet[column] = dataSet[column] / np.max(dataSet['age_processed'])
    return dataSet

def getProcessedData(dataSet):
    normalizeColumn(dataSet, 'age_processed')
    dataSet['class_category'] = dataSet['Pclass'].astype('category')
    return pd.get_dummies(dataSet.loc[:, ['age_processed', 'class_category', 'Sex']]).astype('float32')

def castObjectToCategory(data):
    for i in range(len(data.columns)):
        if data.dtypes[i] == 'object':
            name = data.columns[i]
            data.iloc[:,i] = data.iloc[:,i].astype('category')

data = pd.read_csv('./data/train.csv')

data = data.dropna(subset=['Pclass', 'Age', 'Sex']).reset_index()

females = data[data['Sex'] == 'female'].reset_index()
allMales = data[data['Sex'] == 'male'].reset_index()
males = allMales.truncate(after=len(females))
data = females.append(males, ignore_index = True)
data = data.sample(frac = 1).reset_index(drop = True)
print(data.head())
print('Items: ' + str(len(data)))

castObjectToCategory(data)

print(data.dtypes)
print(data.head(5))

data['y'] = data['Survived'].astype('float32')
data['age_processed'] = data['Age']
processed = getProcessedData(data)
print(processed.head())
print(processed.dtypes)

# quit()

train_idx = pd.Series(np.random.rand(len(data))) > 0.7

x_train = processed[train_idx].as_matrix().astype('float32')
y_train = data[train_idx].as_matrix(['y']).astype('float32')
x_test = processed[~train_idx].as_matrix().astype('float32')
y_test = data[~train_idx].as_matrix(['y']).astype('float32')

x_shape = len(processed.columns)

# Setup model
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(np.zeros([x_shape, 1]), dtype=tf.float32)
b = tf.Variable(0.1, dtype=tf.float32)

y_ = tf.matmul(x,W) + b

loss = tf.reduce_sum(tf.square(y_ - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_len = len(x_train)
loss_historical = []
for i in range(20000):
    idx = i % x_len
    sess.run(train, {x: x_train[idx:idx+1, :], y: y_train[idx:idx+1, :]})
    if i % 500 == 0:
        curr_loss = sess.run(loss, {x: x_test, y: y_test})
        loss_historical.append(curr_loss)
        print(curr_loss)

plt.plot(range(len(loss_historical)), loss_historical)
plt.show()

# quit()

W_res = sess.run(W)

# W_res.describe()
print(W_res)
print(processed.columns)

# ind = plt.arange(processed.columns)
# barchart = plt.bar(ind, W_res, 20)
# plt.plot()

# Validate against known data, not used in training
test_labels = sess.run(y_, {x: x_test}) > 0.5
print(np.mean(np.abs(test_labels - y_test)))

# Predict unknown Kaggle data
verification_set = pd.read_csv('./data/test.csv')
verification_set['age_processed'] = verification_set['Age'].map(lambda e: 30 if np.isnan(e) else e)
vs_processed = getProcessedData(verification_set)
x_vs = vs_processed.as_matrix().astype('float32')

y_vs = sess.run(y_, {x: x_vs})

verification_set['Survived'] = (y_vs > 0.5)
verification_set['Survived'] = verification_set['Survived'].astype('int32')

verification_set.to_csv('./data/kaggle.csv', columns=['PassengerId', 'Survived'], index=False)
quit()


N = len(processed.columns)
men_means = np.transpose(W_res)[0]
print(men_means)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r')

# add some text for labels, title and axes ticks
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(processed.columns)

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

ax.axes.yaxis_inverted = True

# ax.legend([rects1], ['Men'])

plt.show()
