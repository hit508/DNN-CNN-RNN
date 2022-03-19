import os.path

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

new_model = tf.keras.models.load_model('cifar10-model.tf')
new_model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# test_loss, test_acc = new_model.evaluate(x_test, y_test)
# print('evaluate loss: {} - evaluate accuracy: {}'.format(test_loss, test_acc))


def read_image():
    file_name = 'duck.jpg'
    img = tf.io.read_file(filename=file_name)   # 默认读取格式为uint8
    img = tf.image.decode_jpeg(img, channels=0)  # channels 为1得到的是灰度图，为0则按照图片格式来读
    return img


def write_image(img, file_name):
    image = tf.image.encode_jpeg(img)
    tf.io.write_file(os.path.join('.\\images', file_name), image)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(50):
    prediction = []
    for j in range(5):
        test = tf.expand_dims(x_test[i], axis=0)
        prediction.append(np.argmax(new_model.predict(test)))

    index = int(np.mean(prediction))

    if (y_test[i][0] != index):
        print('evaluate type: {} - real type: {}'.format(class_names[index], class_names[y_test[i][0]]))
    else:
        write_image(x_test[i], str(i + 1) + '_' + class_names[index] + '.jpg')


