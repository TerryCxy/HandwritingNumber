from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import os
# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 搭建LeNet网络
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))  # 池化层--平均值
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())  # 将多维转化为一维，卷积层到全连接层过度
    network.add(layers.Dense(84, activation='relu'))  # 全连接神经网络层
    network.add(layers.Dense(10, activation='softmax'))
    return network

network = LeNet()
network.summary()

network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练网络 epochs训练回合 batch_size每次训练数据量
y_pre = network.predict(test_images[:5])
print(y_pre, "\n", test_labels[:5])

network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test loss:", test_loss, "    test accuracy:", test_accuracy)

y_pre = network.predict(test_images[:5])
print(y_pre, "\n", test_labels[:5])

# 保存数据
# network.save('recogNumCNN_model.model')
# print("save successfully")
