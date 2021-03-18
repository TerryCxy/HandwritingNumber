from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
import os

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 将28*28的方阵变成向量
train_images = train_images.reshape((60000, 28*28)).astype('float')
test_images = test_images.reshape((10000, 28*28)).astype('float')
# 1 -> [0 1 0 0 0...]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))  # 正则化 lambda=0.0001
network.add(layers.Dropout(0.01))  # 防止过拟合
network.add(layers.Dense(units=32, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=10, activation='softmax'))

# 优化器：RMSprop   损失函数：交叉熵损失函数
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练网络, epochs训练回合， batch_size每次训练给的数据
network.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)


# 测试模型的性能
y_pre = network.predict(test_images[:1])
print(y_pre, "\n", test_labels[:1])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test loss:", test_loss, "    test accuracy:", test_accuracy)

# 保存数据
# network.save('recogNumFull_model.model')
# print("save successfully")


