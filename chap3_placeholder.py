import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

x1 = tf.placeholder("float", 3)
x2 = tf.placeholder("float", None)
x3 = tf.placeholder("float", 3)
y = x1 * 2 + x2 + x3

x4 = tf.placeholder("float", [None, 3])
y2 = x4 + 1

with tf.Session() as session:
    result = session.run(y, feed_dict={x1: [1, 2, 3], x2: [1, 2, 3], x3: [1, 2, 3] })
    print(result)

    x_data = [[1, 2, 3], [4, 5, 6], ]
    result2 = session.run(y2, feed_dict={x4: x_data})
    print(result2)


# # First, load the image again
# filename = os.path.dirname(__file__) + "/tiger.jpg"
# raw_image_data = mpimg.imread(filename)
#
# #print(raw_image_data)
#
# image = tf.placeholder("uint8", [None, None, 3])
# slice = tf.slice(image, [500, 0, 0], [-1, -1, -1])
#
# with tf.Session() as session:
#     result = session.run(slice, feed_dict={image: raw_image_data})
#
#     print(result.shape)
#
# plt.imshow(result)
# plt.show()