import tensorflow as tf

learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalizing data
'''
X데이터는 흑백이지만, 그 진하기에 따라 0~255까지의 숫자로 되어있다.
0: 흰색, 255: 검정색
이를 0~1사이의 값으로 Nomalize해준다.
'''
x_train, x_test = x_train / 255.0, x_test / 255.0



# change data shape
print(x_train.shape)  # (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) # (60000, 784)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# change result to one-hot encoding
# in tf1, one_hot= True in read_data_sets("MNIST_data/", one_hot=True)
# took care of it, but here we need to manually convert them
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

''' 
original y_train : [5 0 4 ... 5 6 8]
categorical y_train : (one-hot encoding)
[[0. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]]
'''
# # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
# array([0, 2, 1, 2, 0])
# `to_categorical` converts this into a matrix with as many columns as there are classes. The number of rows
#  stays the same. to_categorical(labels)
# array([[ 1.,  0.,  0.],
#        [ 0.,  0.,  1.],
#        [ 0.,  1.,  0.],
#        [ 0.,  0.,  1.],
#        [ 1.,  0.,  0.]], dtype=float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=784, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

predictions = tf.model.predict(x_test)
print('Prediction: \n', predictions)
x_train
score = tf.model.evaluate(x_train, y_train)
print('Accuracy: ', score[1])

"""
print("loss: {0}, accuracy: {1}".format(evaluate[0], evaluate[1]))

600/600 [==============================] - 0s 747us/step - loss: 0.2500 - accuracy: 0.9296
Prediction:
 [[2.3365526e-06 8.5631190e-12 7.2137523e-06 ... 9.9562472e-01
  1.2004549e-05 3.0217407e-04]
 [1.9105324e-04 3.4374762e-06 9.9416566e-01 ... 1.5393519e-17
  3.4459823e-05 5.9198414e-14]
 [1.6588732e-06 9.7850579e-01 1.3058487e-02 ... 1.0039574e-03
  2.8221116e-03 2.3927013e-04]
 ...
 [9.5574393e-09 7.1059953e-09 5.7462175e-06 ... 1.8642319e-03
  5.1601101e-03 1.8177204e-02]
 [1.0611132e-07 2.2056628e-07 2.5900331e-07 ... 9.1484495e-08
  5.5849305e-03 1.3648346e-07]
 [1.4656691e-06 4.4199802e-14 8.9394242e-05 ... 3.5522080e-12
  2.9127078e-08 4.5796957e-11]]
1875/1875 [==============================] - 1s 449us/step - loss: 0.2446 - accuracy: 0.9322
Accuracy:  0.932200014591217
"""