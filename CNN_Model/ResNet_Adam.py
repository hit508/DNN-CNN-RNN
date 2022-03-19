import tensorflow.keras as keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import regularizers
import tensorflow as tf


stack_n = 18
layers = 6 * stack_n + 2
num_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = 128
epochs = 200
iterations = 50000 // batch_size + 1
weight_decay = 1e-4


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def learning_rate_scheduler(epoch, lr):
    if epoch < 160:
        return lr
    return lr * tf.math.exp(-0.1)


def residual_network(img_input, classes_num=10, stack_n=5):
    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


if __name__ == '__main__':
    print("========================================")
    print("MODEL: Residual Network ({:2d} layers)".format(6 * stack_n + 2))
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))

    print("== LOADING DATA... ==")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = residual_network(img_input, num_classes, stack_n)
    resnet = Model(img_input, output)

    # print model architecture if you need.
    # print(resnet.summary())

    # set optimizer
    adam = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1.0e-8, beta_1=0.9, beta_2=0.99)
    resnet.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # set callback
    cbks = [TensorBoard(log_dir='drive/MyDrive/Model/ResNet_Adam/logs', histogram_freq=0),
            LearningRateScheduler(learning_rate_scheduler),
            ModelCheckpoint('drive/MyDrive/Model/ResNet_Adam/logs', mode='auto', save_freq='epoch')]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    data_generator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125,
                                        height_shift_range=0.125, fill_mode='constant', cval=0.)

    data_generator.fit(x_train)

    # start training
    resnet.fit(data_generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
               epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))
    resnet.save('drive/MyDrive/Model/ResNet_Adam/model/resnet.tf')