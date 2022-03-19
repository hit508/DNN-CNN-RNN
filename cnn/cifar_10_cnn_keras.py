import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

batch_size = 128
learning_rate = 0.0002
ratio_grad = 0.9
ratio_momentum = 0.0
decay_regularization = 0.0001
learning_rate_decay = 0.05
max_train_steps = 150
dropout_rate = 0.5
valid_num = 5000

height = 32
width = 32
depth = 3


def learning_rate_scheduler(epoch, lr):
    if epoch < 120:
        return lr
    else:
        return lr * tf.math.exp(-learning_rate_decay)


# 导入数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
x_train.astype('float32')
x_test.astype('float32')

x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

# 数据格式
print("train_images.shape: {}".format(x_train.shape))  # 训练集中有50000张图像，每张图像都为32x32像素
print("train_labels len: {}".format(y_train.shape))  # 训练集中有50000个标签
print("test_images.shape: {}".format(x_test.shape))  # 测试集中有10000张图像，每张图像都为32x32像素
print("test_labels len: {}".format(y_test.shape))  # 测试集中有10000个标签

# 构建模型
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=False,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=False,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                                 data_format='channels_last'))

model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                                 data_format='channels_last'))

model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                              data_format='channels_last', activation='relu', use_bias=True,
                              kernel_initializer=keras.initializers.he_normal(),
                              bias_initializer=keras.initializers.constant(value=0.0),
                              kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                                 data_format='channels_last'))

model.add(keras.layers.Flatten(data_format='channels_last'))
model.add(keras.layers.Dense(units=512, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=keras.initializers.he_normal(),
                             bias_initializer=keras.initializers.constant(value=0.0),
                             kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(dropout_rate))
model.add(keras.layers.Dense(units=384, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=keras.initializers.he_normal(),
                             bias_initializer=keras.initializers.constant(value=0.0),
                             kernel_regularizer=keras.regularizers.L2(decay_regularization)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(dropout_rate))
model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax, use_bias=True,
                             kernel_initializer=keras.initializers.he_normal()))

# 编译模型
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, rho=ratio_grad, momentum=ratio_momentum),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 训练模型
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="demo/logs", histogram_freq=1, update_freq='epoch')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="demo/logs/checkpoint", save_freq='epoch')

learning_rate_callback = keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

data_generator = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125,
                                                              height_shift_range=0.125,
                                                              fill_mode='constant', cval=0.)

data_generator.fit(x_train)

model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size), shuffle=True, verbose=1,
          validation_data=(x_test, y_test),
          epochs=max_train_steps,
          callbacks=[tensorboard_callback, learning_rate_callback, checkpoint_callback])

# 模型保存
model.save('demo/model')
