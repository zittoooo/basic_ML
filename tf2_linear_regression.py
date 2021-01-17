import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# 간단하게 계층을 쌓아 모델링 할 수 있도록 설계되었다.
tf.model = tf.keras.Sequential()

# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2
'''
Sequential 모델을 정의하고 학습하기 전에 .compile() 메서드를 사용하여 학습에 대한 설정을 해줘야한다.
Loss function: 손실함수를 설정해주는 부분이며, 'categorical_crossentropy', 'mse' 처럼 문자열타입으로 설정할 수 있다.
Optimizer: 최적화 함수를 설정하는 부분이며, 'sgd', 'adam', 'rmsprop' 등 문자열타입으로 설정할 수 있다.
Metrics: 모델의 성능을 판정하는데 사용하는 지표 함수이며,['accuracy'] 처럼 리스트 형태로 설정한다.
'''

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)