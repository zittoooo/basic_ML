# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('..\data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6

# Convert y_data to one_hot
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("one_hot:", y_one_hot)
'''
one_hot: [[1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 ...
 ...
'''

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
tf.model.summary()
'''
Model: "sequential_17"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_17 (Dense)             (None, 7)                 119       
=================================================================
Total params: 119
Trainable params: 119
Non-trainable params: 0
'''

history = tf.model.fit(x_data, y_one_hot, epochs=1000)

# Single data test
test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]]) # expected prediction == 3 (feathers)
print(tf.model.predict(test_data), tf.model.predict_classes(test_data))
'''
[[8.7891385e-04 1.3614377e-03 7.1630953e-03 9.8524934e-01 4.0065190e-03
  6.3265378e-07 1.3401081e-03]] [[8.7891385e-04 1.3614377e-03 7.1630953e-03 9.8524934e-01 4.0065190e-03
  6.3265378e-07 1.3401081e-03]]
'''


# Full x_data test
pred = (tf.model.predict(x_data) > 0.5).astype("int32")
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


'''
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False False False False False False False]] Prediction: [0 0 0 1 0 0 0] True Y: 3
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False False False False False False False]] Prediction: [0 0 0 1 0 0 0] True Y: 3
[[False False False False False False False]] Prediction: [0 0 0 1 0 0 0] True Y: 3
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True  True  True  True  True  True]] Prediction: [1 0 0 0 0 0 0] True Y: 0
[[False  True False False False False False]] Prediction: [0 1 0 0 0 0 0] True Y: 1
[[False False False False False False False]] Prediction: [0 0 0 1 0 0 0] True Y: 3
[[False False False False False False False]] Prediction: [0 0 0 0 0 0 1] True Y: 6
...
...

'''