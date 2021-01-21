# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

x_raw = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_raw = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

nb_classes = 3

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=4, units=nb_classes, use_bias=True))  # use_bias is True, by default
#use_bias  편향, 예측값과 정답이 대체로 멀리 떨어져 있으면 결과의 편향이 높다고 말한다.

# use softmax activations: softmax = exp(logits) / reduce_sum(exp(logits), dim)
tf.model.add(tf.keras.layers.Activation('softmax'))

# use loss == categorical_crossentropy
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
tf.model.summary()
'''
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_7 (Dense)              (None, 3)                 15        
_________________________________________________________________
activation_2 (Activation)    (None, 3)                 0         
=================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
__________________________
'''

history = tf.model.fit(x_data, y_data, epochs=2000)

'''
Epoch 999/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2402 - accuracy: 1.0000
Epoch 1000/1000
1/1 [==============================] - 0s 13ms/step - loss: 0.2401 - accuracy: 1.0000
'''

print('--------------')
# Testing & One-hot encoding
a = tf.model.predict(np.array([[1, 11, 7, 9]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))  # axis=1 행에서 최대값을 찾는다
'''
[[6.6087306e-02 9.3386191e-01 5.0791976e-05]] [1]
'''

print('--------------')
b = tf.model.predict(np.array([[1, 3, 4, 3]]))
print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))
'''
[[0.71742517 0.239744   0.04283078]] [0]
'''


print('--------------')
# or use argmax embedded method
c = tf.model.predict(np.array([[1, 1, 0, 1]]))
c_onehot = (tf.model.predict(np.array([[1, 1, 0, 1]])) > 0.5).astype("int32")
print(c, c_onehot)
'''
[[4.3300852e-06 3.4603288e-03 9.9653530e-01]] [[0 0 1]]
'''

print('--------------')
all = tf.model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
all_onehot = (tf.model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])) > 0.5).astype("int32")
print(all, all_onehot)

'''
[[4.4172323e-01 5.5817616e-01 1.0059154e-04]
 [6.0448396e-01 3.1945682e-01 7.6059170e-02]
 [6.3140978e-06 3.6897785e-03 9.9630392e-01]] 
 [[0 1 0]
 [1 0 0]
 [0 0 1]]
'''