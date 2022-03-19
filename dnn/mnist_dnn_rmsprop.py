import tensorflow as tf
import tensorflow.keras as keras

batch_num = 100
batch_size = 600
epoch = 20
decay_regularization = 0.0001

ratio_momentum = 0.2
ratio_gradient = 0.99
rmsprop_deta = 1.0e-8
ratio_polyak = 0.9

learning_rate = 0.001
decay_steps = 100
learning_rate_decay = 0.9

def exponential_decay(learning_rate, global_step, decay_steps, learning_rate_decay):
    step = tf.cast(global_step // decay_steps, tf.float32)
    return learning_rate * tf.math.pow(learning_rate_decay, step)

#定义前向传播函数,输出未归一化
def forward_prop(input_tensor, train_variable):
    w1 = tf.reshape(tf.slice(train_variable, begin=[0], size=[784 * 500]), shape=[784, 500])
    b1 = tf.reshape(tf.slice(train_variable, begin=[784 * 500], size=[500]), shape=[500])
    w2 = tf.reshape(tf.slice(train_variable, begin=[784 * 500 + 500], size=[500 * 10]), shape=[500, 10])
    b2 = tf.reshape(tf.slice(train_variable, begin=[784 * 500 + 500 + 500 * 10], size=[10]), shape=[10])
    hidden_layer = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
    return tf.matmul(hidden_layer, w2)+b2


def read_image():
    img_list = []
    for i in range(10):
        file_name = '../img/' + str(i) + '.jpg'
        img = tf.io.read_file(filename=file_name)  # 默认读取格式为uint8
        img = tf.image.decode_jpeg(img, channels=1)  # channels 为1得到的是灰度图，为0则按照图片格式来读
        img = 255 - img
        img_list.append(img)
    return img_list

#导入数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("train_images.shape: {}".format(x_train.shape))  # 训练集中有60000张图像，每张图像都为28x28像素
print("train_labels len: {}".format(len(y_train)))  # 训练集中有60000个标签
print("train_labels: {}".format(y_train))  # 每个标签都是一个介于 0 到 9 之间的整数
print("test_images.shape: {}".format(x_test.shape))  # 测试集中有10000张图像，每张图像都为28x28像素
print("test_labels len: {}".format(len(y_test)))  # 测试集中有10000个标签
print("test_labels: {}".format(y_test))

#输入数据归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

#生成隐藏层参数，其中weights包含784x500=392000个参数
weights1 = tf.Variable(tf.random.truncated_normal([784, 500], stddev=0.1), dtype=tf.float32)
biases1 = tf.Variable(tf.constant(0.1, shape=[500]))

#生成输出层参数，其中weights2包含500x10=5000个参数
weights2 = tf.Variable(tf.random.truncated_normal([500, 10], stddev=0.1), dtype=tf.float32)
biases2 = tf.Variable(tf.constant(0.1, shape=[10]))


v = tf.Variable(tf.zeros(shape=[784 * 500 + 500 * 10 + 500 + 10]), trainable=False)
r = tf.Variable(tf.zeros(shape=[784 * 500 + 500 * 10 + 500 + 10]), trainable=False)

train_variable_avg = tf.Variable(tf.zeros(shape=[784 * 500 + 500 * 10 + 500 + 10]), trainable=False)

train_variable = tf.concat([tf.reshape(weights1, [-1]), tf.reshape(biases1, [-1]),
    tf.reshape(weights2, [-1]), tf.reshape(biases2, [-1])], axis=0)
print("train_variable.shape: {}".format(train_variable.shape))


for k in range(epoch):
    print("train_epoch: {}".format(k))
    tf.random.shuffle(x_train)
    x_train_batch = tf.convert_to_tensor(tf.split(x_train, num_or_size_splits=batch_num, axis=0), dtype=tf.float32)
    print("x_train_batch.shape: {}".format(x_train_batch.shape))
    y_train_batch = tf.convert_to_tensor(tf.split(y_train, num_or_size_splits=batch_num, axis=0), dtype=tf.int32)
    print("y_train_batch.shape: {}".format(y_train_batch.shape))
    for train_step in range(batch_num):
        # 输入输出
        image = tf.reshape(x_train_batch[train_step], shape=[batch_size, 784])
        label = tf.one_hot(indices=y_train_batch[train_step], depth=10)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(train_variable)
            # 交叉熵
            train_variable = train_variable + ratio_momentum * v
            y_predict = forward_prop(image, train_variable)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_predict)
            # 正则化项
            w1 = tf.reshape(tf.slice(train_variable, begin=[0], size=[784 * 500]), shape=[784, 500])
            w2 = tf.reshape(tf.slice(train_variable, begin=[784 * 500 + 500], size=[500 * 10]), shape=[500, 10])
            regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
            total_loss = tf.reduce_mean(cross_entropy) + decay_regularization * regularization

        # SGD梯度计算
        grad = tape.gradient(total_loss, train_variable)

        # RMSProp自适应学习率
        r = ratio_gradient * r + (1 - ratio_gradient) * tf.multiply(grad, grad)
        v = ratio_momentum * v - learning_rate * grad / (tf.math.sqrt(r) + rmsprop_deta)

        train_variable = train_variable + v
        tf.nn.dropout(train_variable, rate=0.5, seed=1)

        #Polyak平均
        train_variable_avg = ratio_polyak * train_variable_avg + (1.0 - ratio_polyak) * train_variable

    image_train = tf.cast(tf.reshape(x_train, shape=[60000, 784]), tf.float32)
    y_train_predict = tf.nn.softmax(forward_prop(image_train, train_variable_avg), axis=1)
    crorent_train = tf.math.equal(tf.argmax(y_train_predict, 1), y_train)
    train_accuracy = tf.reduce_mean(tf.cast(crorent_train, tf.float32))
    print("train accuracy: {}%".format(train_accuracy * 100))

image = tf.cast(tf.reshape(x_test, shape=[10000, 784]), tf.float32)
y_test_predict = tf.nn.softmax(forward_prop(image, train_variable_avg), axis=1)
crorent_predicition = tf.math.equal(tf.argmax(y_test_predict, 1), y_test)
accuracy = tf.reduce_mean(tf.cast(crorent_predicition, tf.float32))
print("test accuracy: {}%".format(accuracy * 100))

test_img_list = read_image()
test_img = tf.convert_to_tensor(test_img_list, dtype=tf.float32)
test_img = test_img / 255.0
test_img = tf.reshape(test_img, [10, 784])
test_predict = tf.nn.softmax(forward_prop(test_img, train_variable), axis=1)
print(tf.argmax(test_predict, axis=1))