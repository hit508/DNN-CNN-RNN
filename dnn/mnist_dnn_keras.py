import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

batch_size = 100
max_tain_steps = 10

#导入数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#输入数据归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 数据格式
print("train_images.shape: {}".format(x_train.shape))  # 训练集中有60000张图像，每张图像都为28x28像素
print("train_labels len: {}".format(len(y_train)))  # 训练集中有60000个标签
print("train_labels: {}".format(y_train))  # 每个标签都是一个介于 0 到 9 之间的整数
print("test_images.shape: {}".format(x_test.shape))  # 测试集中有10000张图像，每张图像都为28x28像素
print("test_labels len: {}".format(len(y_test)))  # 测试集中有10000个标签
print("test_labels: {}".format(y_test))

#构建模型
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(units=500, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))

#编译模型
model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#训练模型
model.fit(x_train, y_train, batch_size, epochs=max_tain_steps, verbose=2)

#模型评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss: {} - Test accuracy: {}'.format(test_loss, test_acc))

#模型预测
predictions = model.predict(x_test)
print("The first prediction: {}".format(np.argmax(predictions[0])))