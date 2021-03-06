import tensorflow.keras as keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import regularizers

# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER',
#                     help='batch size(default: 128)')
# parser.add_argument('-e', '--epochs', type=int, default=200, metavar='NUMBER',
#                     help='epochs(default: 200)')
# parser.add_argument('-n', '--stack_n', type=int, default=5, metavar='NUMBER',
#                     help='stack number n, total layers = 6 * n + 2 (default: 5)')
# parser.add_argument('-d', '--dataset', type=str, default="cifar10", metavar='STRING',
#                     help='dataset. (default: cifar10)')


stack_n = 8
layers = 6 * stack_n + 2
num_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = 128
epochs = 200
iterations = 50000 // batch_size + 1
weight_decay = 1e-4
learning_rate_decay = 0.05

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def learningrate_scheduler(epoch, lr):
    if epoch < 90:
        return lr
    if epoch < 122:
        return lr * 0.1
    return lr *0.01


def adam_learningrate_scheduler(epoch, lr):
    if epoch < 90:
        return lr
    if epoch < 124:
        return lr * learning_rate_decay
    return lr * learning_rate_decay * learning_rate_decay


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
    print(resnet.summary())

    # set optimizer
    #sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.99)
    resnet.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # set callback
    cbks = [TensorBoard(log_dir='./ResNet/logs', histogram_freq=0),
            LearningRateScheduler(adam_learningrate_scheduler),
            ModelCheckpoint('./ResNet/logs', mode='auto', save_freq='epoch')]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    data_generator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125,
                                        height_shift_range=0.125, fill_mode='constant', cval=0.)

    data_generator.fit(x_train)

    # start training
    resnet.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    resnet.save('./ResNet/model/resnet.tf')
