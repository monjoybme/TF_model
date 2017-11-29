import tensorflow as tf
import time
import sys
from tensorflow.python import debug as tf_debug
import numpy as np

# Parameters

training_epochs = 10
display_step = 1
batch_size = 10
width = 200
height = 200
#number of classes
nClass = 2
addr_train = "/home/pavelkrolevets/Working/TF_facenet/data/output/train-00000-of-00001"
addr_val = "/home/pavelkrolevets/Working/TF_facenet/data/output/validation-00000-of-00001"

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# Counting number of samples in TFrecord
def rec_iter (FILE):
    i = sum(1 for _ in tf.python_io.tf_record_iterator(FILE))
    return i

N_OF_SAMPL_TRAIN = rec_iter(addr_train)
N_OF_SAMPL_VAL = rec_iter(addr_val)
print('Number of samples for training', N_OF_SAMPL_TRAIN)
print('Number of samples for validating', N_OF_SAMPL_VAL)
# Function to tell TensorFlow how to read a single image from input file
def getImage(filenameQ):

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']
    file = features['image/filename']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel.
    # the "1-.." part inverts the image, so that the background is black.

    image = tf.reshape(image, [width, height, 3])

    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes.
    label = tf.stack(tf.one_hot(label - 1, nClass))
    #label = tf.to_int32(label-1)

    return label, image, file




#MODEL

# Architecture

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b, name='last_layer')


def inference(x, keep_prob):

    #x = tf.reshape(x, shape=[-1, height, width, 3])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 32], [32])

    with tf.variable_scope('max_pool1'):
        pool_1 = max_pool(conv_1)

    with tf.variable_scope('batch_norm'):
        norm1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(norm1, [5, 5, 32, 64], [64])

    with tf.variable_scope('max_pool2'):
        pool_2 = max_pool(conv_2)

    with tf.variable_scope('batch_norm'):
            norm2 = tf.nn.lrn(pool_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    with tf.variable_scope("conv_3"):
        conv_3 = conv2d(norm2, [5, 5, 64, 32], [32])

    with tf.variable_scope('max_pool3'):
        pool_3 = max_pool(conv_3)

    with tf.variable_scope("fc1"):
        pool_2_flat = tf.reshape(pool_3, [-1, 25 * 25 * 32])
        fc_1 = layer(pool_2_flat, [25*25*32, 384], [384])
    with tf.variable_scope("fc2"):
        fc_2 = layer(fc_1, [384, 192], [192])
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_2, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_1_drop, [192, nClass], [nClass])

    return output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                               1000, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op



def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    print(correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy



# import data
# associate the "label" and "image" objects with the corresponding features read from
# a single example in the training data file
# convert filenames to a queue for an input pipeline.

filenameQ_train = tf.train.string_input_producer([addr_train], num_epochs=None)
label, image, file = getImage(filenameQ_train)

print(label, '\n', image)
# similarly for the validation data

filenameQ_val = tf.train.string_input_producer([addr_val], num_epochs=None)
vlabel, vimage, vfile = getImage(filenameQ_val)

# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
# of labels and images respectively
imageBatch, labelBatch, fileBatch = tf.train.shuffle_batch(
    [image, label, file], batch_size=batch_size,
    capacity=2000,
    min_after_dequeue=200)

# and similarly for the validation data
vimageBatch, vlabelBatch, vfileBatch = tf.train.shuffle_batch(
    [vimage, vlabel, vfile], batch_size=batch_size,
    capacity=2000,
    min_after_dequeue=200)


# # DEBUG
# sess = tf.InteractiveSession()
# addr_train = "/home/pavelkrolevets/Working/TF_facenet/data/output/train-00000-of-00001"
# # filenameQ_test = tf.train.string_input_producer([addr_test], num_epochs=1)
# # label, image = getImage(filenameQ_test)
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#
# print(v1)
# sess.close()


x = tf.placeholder("float", [None, height, width, 3], name='placehold_x')
y = tf.placeholder('float', [None, nClass], name='placehold_y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout probability

output = inference(x, keep_prob)
cost = loss(output, y)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Passing global_step to minimize() will increment it at each step.

train_op = training(cost, global_step)
eval_op = evaluate(output, y)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()



#top_k_op = tf.nn.in_top_k(output, y, 1)


sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)
summary_writer = tf.summary.FileWriter("summary_logs/", graph_def=sess.graph_def)
init_op = tf.global_variables_initializer()
sess.run(init_op)

#train_eval_op = evaluate(output, y)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)



with tf.device('/gpu:0'):
    with sess.as_default():
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch_train = int(N_OF_SAMPL_TRAIN/batch_size)

            # Loop over all batches
            for i in range(total_batch_train):
                minibatch_x, minibatch_y = sess.run([imageBatch, labelBatch])
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                #train_precision = sess.run(eval_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 1})

                # Compute average loss
                #avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/batch_size
                print('Epoch #: ', epoch, '  Batch #: ', i) #, 'loss: ', avg_cost, 'train precision: ', (1-train_precision))

            # Display logs per epoch step
            if epoch % display_step == 0:

                vbatch_xs, vbatch_ys, vbatch_zs = sess.run([vimageBatch, vlabelBatch, vfileBatch])
                accuracy = sess.run(eval_op, feed_dict={x: vbatch_xs, y: vbatch_ys, keep_prob: 1})
                prediction = sess.run(output, feed_dict={x: vbatch_xs, keep_prob: 1})

                print ("Validation Error:", 1-accuracy)
                print (prediction, vbatch_zs)

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 1})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                saver.save(sess, "model_logs/model-checkpoint", global_step=global_step)


        print ("Optimization Finished!")

        # total_batch_eval=N_OF_SAMPL_VAL/batch_size
        #
        # for i in range(total_batch_eval):
        #     vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        #     accuracy = sess.run(eval_op, feed_dict={x: vimageBatch, y: vlabelBatch, keep_prob: 1})
        #
        # print ("Test Accuracy:", accuracy)

        # finalise
        coord.request_stop()
        coord.join(threads)

### Predict

img_addr = '/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/dog/dog.155.jpg'
with tf.gfile.FastGFile(img_addr, 'rb') as f:
    image_data = f.read()
image = tf.image.decode_jpeg(image_data, channels=3)
image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
image = tf.reshape(image, [-1, width, height, 3])
pred_im = sess.run(image)

prediction = sess.run(inference, feed_dict={x: pred_im, keep_prob: 1})
print(prediction)


