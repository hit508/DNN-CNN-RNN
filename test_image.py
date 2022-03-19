import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def read_image():
    file_name = 'duck.jpg'
    img = tf.io.read_file(filename=file_name)   # 默认读取格式为uint8
    img = tf.image.decode_jpeg(img, channels=0)  # channels 为1得到的是灰度图，为0则按照图片格式来读
    return img

def write_image(img):
    file_name = 'duck-copy.jpg'
    imge = tf.image.encode_jpeg(img)
    tf.io.write_file(file_name, imge)

def main():
    with tf.device('GPU:0'):
        img = read_image()
        print(img)
        print(img.dtype)
        print(img.shape)
        #plt.imshow(img)
        #plt.show()
        #img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_brightness(img, max_delta=1)
        #img = tf.image.random_contrast(img, 0.2, 10)
        #img = tf.image.adjust_saturation(img, 2)
        #img = tf.image.per_image_standardization(img)
        #img = tf.image.resize(img, [1000, 1200], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #img = tf.image.resize_with_crop_or_pad(img, 300, 300)
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.8], [0.2, 0.3, 0.5, 0.5]]])
        (begin, size, bboxes) = tf.image.sample_distorted_bounding_box(tf.shape(img), boxes)
        img_batched = tf.expand_dims(tf.image.convert_image_dtype(img, tf.float32), axis=0)
        color = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        img_batched = tf.image.draw_bounding_boxes(img_batched, bboxes, color)
        plt.imshow(img_batched[0])
        plt.show()
        slice_img = tf.slice(img, begin, size)
        plt.imshow(slice_img)
        plt.show()
        #write_image(img)

if __name__ == "__main__":
    main()