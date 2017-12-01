import tensorflow as tf
import time
import sys
from tensorflow.python import debug as tf_debug
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters

training_epochs = 10
display_step = 1
batch_size = 100
width = 28
height = 28
#number of classes
nClass = 10

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.

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

    x = tf.reshape(x, shape=[-1, height, width, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])

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

    # with tf.variable_scope('max_pool3'):
    #     pool_3 = max_pool(conv_3)

    with tf.variable_scope("fc1"):
        pool_2_flat = tf.reshape(conv_3, [-1, 7 * 7 * 32])
        fc_1 = layer(pool_2_flat, [7*7*32, 384], [384])
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
                                               100, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op



def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy

x = tf.placeholder("float", [None, height* width*1], name='placehold_x')
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

sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)
summary_writer = tf.summary.FileWriter("summary_logs/", graph_def=sess.graph_def)
init_op = tf.global_variables_initializer()
sess.run(init_op)

with tf.device('/gpu:0'):
    with sess.as_default():
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch_train = int(mnist.train.num_examples / batch_size)

            # Loop over all batches
            for i in range(total_batch_train):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                #print (minibatch_x, 'n/', minibatch_y)
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                train_precision = sess.run(eval_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 1})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/batch_size
                if not i % 10:
                    print('Epoch #: ', epoch, '  Batch #: ', i, 'loss: ', avg_cost, 'train error: ', (1-train_precision))

                # Display logs per epoch step
            if epoch % display_step == 0:
                minibatch_x_val, minibatch_y_val = mnist.validation.next_batch(batch_size)
                accuracy = sess.run(eval_op,
                                    feed_dict={x: minibatch_x_val, y: minibatch_y_val, keep_prob: 1})
                print("Validation Error:", (1 - accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 1})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                saver.save(sess, "model_logs/model-checkpoint", global_step=global_step)

                prediction = sess.run(output, feed_dict={x: minibatch_x_val, keep_prob: 1})
                print(prediction)

        print ("Optimization Finished!")


        # total_batch_eval=N_OF_SAMPL_VAL/batch_size
        #
        # for i in range(total_batch_eval):
        #     vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        #     accuracy = sess.run(eval_op, feed_dict={x: vimageBatch, y: vlabelBatch, keep_prob: 1})
        #
        # print ("Test Accuracy:", accuracy)

        # finalise
        # coord.request_stop()
        # coord.join(threads)

### Predict
#
# img_addr = '/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/dog/dog.155.jpg'
# with tf.gfile.FastGFile(img_addr, 'rb') as f:
#     image_data = f.read()
# image = tf.image.decode_jpeg(image_data, channels=3)
# image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
# image = tf.reshape(image, [-1, width, height, 3])
# pred_im = sess.run(image)
#
# prediction = sess.run(inference, feed_dict={x: pred_im, keep_prob: 1})
# print(prediction)


