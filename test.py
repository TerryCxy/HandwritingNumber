from keras.models import load_model
from keras.datasets import mnist
import cv2

network = load_model('recogNumFull_model.model')
network.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

image = cv2.imread("six.jpg")
shrink = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imwrite("six_reshape.jpg", shrink)
