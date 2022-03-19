import cifar_10_data
import tensorflow as tf
import tensorflow.keras as keras

batch_size = 128
data_dir = "./cifar-10-batches-py"
epoch = 20
decay_regularization = 0.0004
ratio_polyak = 0.98
learning_rate = 0.0001
ratio_grad = 0.9


def train_image_process(image, label):
    print("image.shape: {}".format(image.shape))
    image = tf.reshape(image, [cifar_10_data.depth, cifar_10_data.height, cifar_10_data.width])
    print("image.shape: {}".format(image.shape))
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
    cropped_image = tf.image.random_crop(image, [24, 24, 3])
    flipped_image = tf.image.random_flip_left_right(cropped_image)
    adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
    adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(adjusted_contrast)
    float_image.set_shape([24, 24, 3])

    return float_image, label


def test_image_process(image):
    image = tf.reshape(image, [cifar_10_data.depth, cifar_10_data.height, cifar_10_data.width])
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
    resized_image = tf.image.resize_with_crop_or_pad(image, 24, 24)
    float_image = tf.image.per_image_standardization(resized_image)
    float_image.set_shape([24, 24, 3])
    return float_image


def variable_with_weight_loss(shape, stddev):
    var = tf.Variable(tf.random.truncated_normal(shape, stddev=stddev), trainable=True)
    return var


def forward_prop(image, kernel1, kernel2, weight1, weight2, weight3, bias1, bias2, fc_bias1, fc_bias2, fc_bias3):
    conv1 = tf.nn.conv2d(image, kernel1, strides=[1, 1, 1, 1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    reshape = tf.reshape(pool2, [batch_size, -1])
    fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
    fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)
    label_predict = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

    return label_predict


def loss_function(label, label_predict, weight1, weight2):
    # label = tf.one_hot(indices=label, depth=10)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=label_predict, labels=label)
    weight_loss = tf.math.multiply(tf.nn.l2_loss(weight1), decay_regularization) + \
                  tf.math.multiply(tf.nn.l2_loss(weight2), decay_regularization)
    loss = tf.reduce_mean(cross_entropy) + weight_loss
    return loss


def main():
    train_data, train_label = cifar_10_data.get_train_data(data_dir=data_dir)
    test_data, test_label = cifar_10_data.get_test_data(data_dir=data_dir)
    eval_data, eval_label = cifar_10_data.get_eval_data(data_dir=data_dir)
    print("train_data.shape: {}".format(train_data.shape))
    print("train_label.shape: {}".format(train_label.shape))
    print("test_data.shape: {}".format(test_data.shape))
    print("test_label.shape: {}".format(test_label.shape))
    print("eval_data.shape: {}".format(eval_data.shape))
    print("eval_label.shape: {}".format(eval_label.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).map(train_image_process)
    eval_data = tf.cast(eval_data, tf.float32)
    eval_data = tf.map_fn(test_image_process, eval_data)
    test_data = tf.cast(test_data, tf.float32)
    test_data = tf.map_fn(test_image_process, test_data)

    kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2)
    kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2)
    weight1 = variable_with_weight_loss(shape=[2304, 384], stddev=0.04)
    weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04)
    weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0)

    bias1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True)
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]), trainable=True)
    fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]), trainable=True)
    fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]), trainable=True)
    fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]), trainable=True)

    avg_kernel1 = tf.Variable(tf.zeros(shape=[5, 5, 3, 64]), trainable=False)
    avg_kernel2 = tf.Variable(tf.zeros(shape=[5, 5, 64, 64]), trainable=False)
    avg_weight1 = tf.Variable(tf.zeros(shape=[2304, 384]), trainable=False)
    avg_weight2 = tf.Variable(tf.zeros(shape=[384, 192]), trainable=False)
    avg_weight3 = tf.Variable(tf.zeros(shape=[192, 10]), trainable=False)

    avg_bias1 = tf.Variable(tf.zeros(shape=[64]), trainable=False)
    avg_bias2 = tf.Variable(tf.zeros(shape=[64]), trainable=False)
    avg_fc_bias1 = tf.Variable(tf.zeros(shape=[384]), trainable=False)
    avg_fc_bias2 = tf.Variable(tf.zeros(shape=[192]), trainable=False)
    avg_fc_bias3 = tf.Variable(tf.zeros(shape=[10]), trainable=False)

    for k in range(epoch):
        print("train_epoch: {}".format(k))
        train_dataset.shuffle(batch_size, reshuffle_each_iteration=True)
        train_batch = train_dataset.batch(batch_size, drop_remainder=True)
        batch_num = train_batch.cardinality().numpy()
        # print("train_batch.cardinality: {}".format(batch_num))

        for batch_image, batch_label in train_batch:
            # print("batch_image.shape: {}".format(batch_image.shape))
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=ratio_grad)
            with tf.GradientTape(persistent=True) as tape:
                label_predict = forward_prop(batch_image, kernel1, kernel2, weight1, weight2, weight3,
                                             bias1, bias2, fc_bias1, fc_bias2, fc_bias3)
                total_loss = loss_function(batch_label, label_predict, weight1, weight2)

            train_var = [kernel1, kernel2, weight1, weight2, weight3, bias1, bias2, fc_bias1, fc_bias2, fc_bias3]
            grads = tape.gradient(total_loss, train_var)
            opt.apply_gradients(zip(grads, train_var))

            avg_kernel1 = avg_kernel1 * ratio_polyak + (1.0 - ratio_polyak) * kernel1
            avg_kernel2 = avg_kernel2 * ratio_polyak + (1.0 - ratio_polyak) * kernel2
            avg_weight1 = avg_weight1 * ratio_polyak + (1.0 - ratio_polyak) * weight1
            avg_weight2 = avg_weight2 * ratio_polyak + (1.0 - ratio_polyak) * weight2
            avg_weight3 = avg_weight3 * ratio_polyak + (1.0 - ratio_polyak) * weight3

            avg_bias1 = avg_bias1 * ratio_polyak + (1.0 - ratio_polyak) * bias1
            avg_bias2 = avg_bias2 * ratio_polyak + (1.0 - ratio_polyak) * bias2
            avg_fc_bias1 = avg_fc_bias1 * ratio_polyak + (1.0 - ratio_polyak) * fc_bias1
            avg_fc_bias2 = avg_fc_bias2 * ratio_polyak + (1.0 - ratio_polyak) * fc_bias2
            avg_fc_bias3 = avg_fc_bias3 * ratio_polyak + (1.0 - ratio_polyak) * fc_bias3

        eval_predict = forward_prop(eval_data, avg_kernel1, avg_kernel2, avg_weight1, avg_weight2, avg_weight3,
                                    avg_bias1, avg_bias2, avg_fc_bias1, avg_fc_bias2, avg_fc_bias3)
        eval_predict = tf.nn.softmax(eval_predict, axis=1)
        accuracy = tf.math.equal(tf.cast(tf.argmax(eval_predict, 1), tf.int64), tf.cast(eval_label, tf.int64))
        train_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        print("train accuracy: {}%".format(train_accuracy * 100))
        eval_loss = loss_function(eval_label, eval_predict, avg_weight1, avg_weight2)
        print("eval loss: {}".format(eval_loss))

    test_predict = forward_prop(test_data, avg_kernel1, avg_kernel2, avg_weight1, avg_weight2, avg_weight3,
                                avg_bias1, avg_bias2, avg_fc_bias1, avg_fc_bias2, avg_fc_bias3)
    test_predict = tf.nn.softmax(test_predict, axis=1)
    accuracy = tf.math.equal(tf.cast(tf.argmax(test_predict, 1), tf.int64), tf.cast(test_label, tf.int64))
    test_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    print("test accuracy: {}%".format(test_accuracy * 100))


if __name__ == '__main__':
    with tf.device('GPU:0'):
        main()
