import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph = tf.Graph()
with graph.as_default():
    matrix = tf.constant([
        [0, 0, 1, 7],
        [0, 2, 0, 0],
        [5, 2, 0, 0],
        [0, 0, 9, 8],
    ])
    reshaped = tf.cast(tf.reshape(matrix, (1, 4, 4, 1)), tf.float32)
    tf.nn.avg_pool(reshaped, ksize=2, strides=2, padding="SAME")

writer = tf.summary.create_file_writer("logs")
with writer.as_default():
    tf.summary.graph(graph)
