import tensorflow as tf
import time
import sys
from tensorflow.python import debug as tf_debug
import numpy as np

width = 100
height = 100
nClass = 2
batch_size = 10
addr_val = "/home/pavelkrolevets/Working/TF_facenet/data/output/validation-00000-of-00001"

# def getImage(filenameQ):
#
#     # object to read records
#     recordReader = tf.TFRecordReader()
#
#     # read the full set of features for a single example
#     key, fullExample = recordReader.read(filenameQ)
#
#     # parse the full example into its' component features.
#     features = tf.parse_single_example(
#         fullExample,
#         features={
#             'image/height': tf.FixedLenFeature([], tf.int64),
#             'image/width': tf.FixedLenFeature([], tf.int64),
#             'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#             'image/channels': tf.FixedLenFeature([], tf.int64),
#             'image/class/label': tf.FixedLenFeature([], tf.int64),
#             'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#             'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#             'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#             'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
#         })
#
#     # now we are going to manipulate the label and image features
#
#     label = features['image/class/label']
#     image_buffer = features['image/encoded']
#     file = features['image/filename']
#     # Decode the jpeg
#     with tf.name_scope('decode_jpeg', [image_buffer], None):
#         # decode
#         image = tf.image.decode_jpeg(image_buffer, channels=3)
#
#         # and convert to single precision data type
#         image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#
#     # cast image into a single array, where each element corresponds to the greyscale
#     # value of a single pixel.
#     # the "1-.." part inverts the image, so that the background is black.
#
#     image = tf.reshape(image, [width, height, 3])
#
#     # re-define label as a "one-hot" vector
#     # it will be [0,1] or [1,0] here.
#     # This approach can easily be extended to more classes.
#     label = tf.stack(tf.one_hot(label - 1, nClass))
#
#     return label, image, file
#
# filenameQ_train = tf.train.string_input_producer([addr_val], num_epochs=None)
# label, image, file = getImage(filenameQ_train)
#
# imageBatch, labelBatch, fileBatch = tf.train.shuffle_batch(
#     [image, label, file], batch_size=batch_size,
#     capacity=2000,
#     min_after_dequeue=200)
#
# sess=tf.Session()
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# minibatch_x, minibatch_y, minibatch_z = sess.run([imageBatch, labelBatch, fileBatch])
#
# coord.request_stop()
# coord.join(threads)

img_addr = ['/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/cat/cat.223.jpg',
            '/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/dog/dog.57.jpg']
checkpoint_directory = '/home/pavelkrolevets/Working/TF_facenet/model_logs/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_directory)

#session_conf = tf.ConfigProto(allow_safe_placement=True, log_device_placement=False)

graph=tf.Graph()

with tf.Session(graph=graph) as sess:

    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('placehold_x:0')
    # keep_prob = graph.get_tensor_by_name('keep_prob:0')  # dropout probability
    oper_restore = graph.get_tensor_by_name('inference:0')

    for img in img_addr:
        with tf.gfile.FastGFile(img, 'rb') as f:
            image_data = f.read()

        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        image = tf.reshape(image, [-1, width, height, 3])
        im_pred = image.eval()

        prediction = sess.run(oper_restore, feed_dict={x: im_pred})
        prediction = sess.run(tf.nn.softmax(prediction))
        print(prediction)#, '\n', minibatch_y, '\n', minibatch_z)


    # prediction = sess.run(oper_restore, feed_dict={x: minibatch_x})
    # prediction = sess.run(tf.nn.softmax(prediction))
    # print(prediction, '\n', minibatch_y, '\n', minibatch_z)




