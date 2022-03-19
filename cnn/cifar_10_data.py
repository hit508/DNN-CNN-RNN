import os

import numpy as np
import tensorflow as tf
import pickle

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

# 数据集参数
label_bytes = 1
height = 32
width = 32
depth = 3
batch_num = 10000
batch_size = 128

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def get_train_data(data_dir):
    filenames = [os.path.join(data_dir, "data_batch_%d" % i) for i in range(1, 5)]
    train_data_list = []
    train_label_list = []
    batch = 0
    for file in filenames:
        data_dict = unpickle(file)
        train_data_list.append(data_dict[b'data'])
        train_label_list.append(data_dict[b'labels'])
        batch += 1
    total_num = batch * batch_num
    train_data_list = tf.reshape(train_data_list, [total_num, height * width * depth])
    train_label_list = tf.reshape(train_label_list, [total_num])
    return train_data_list, train_label_list


def get_test_data(data_dir):
    filename = os.path.join(data_dir, "test_batch")
    data_dict = unpickle(filename)
    test_data_list = tf.reshape(data_dict[b'data'], [batch_num, height * width * depth])
    test_data_list = tf.slice(test_data_list, begin=[0, 0], size=[batch_size, height * width * depth])
    test_label_list = tf.reshape(data_dict[b'labels'], [batch_num])
    test_label_list = tf.slice(test_label_list, begin=[0], size=[batch_size])
    return test_data_list, test_label_list


def get_eval_data(data_dir):
    filename = os.path.join(data_dir, "eval_batch")
    data_dict = unpickle(filename)
    eval_data_list = tf.reshape(data_dict[b'data'], [batch_num, height * width * depth])
    eval_data_list = tf.slice(eval_data_list, begin=[0, 0], size=[batch_size, height * width * depth])
    eval_label_list = tf.reshape(data_dict[b'labels'], [batch_num])
    eval_label_list = tf.slice(eval_label_list, begin=[0], size=[batch_size])
    return eval_data_list, eval_label_list

