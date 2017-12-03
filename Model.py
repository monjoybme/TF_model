import tensorflow as tf
import time
import re
import sys
from tensorflow.python import debug as tf_debug
import numpy as np

# Parameters

training_epochs = 10
display_step = 1
batch_size = 32
width = 100
height = 100
#number of classes
nClass = 2
addr_train = "/home/pavelkrolevets/Working/TF_facenet/data/output/train-00000-of-00001"
addr_val = "/home/pavelkrolevets/Working/TF_facenet/data/output/validation-00000-of-00001"


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
TOWER_NAME = 'tower'

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

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


#MODEL

# Architecture

def conv2d(input, weight_shape, bias_shape, stdev, wd):
    dtype = tf.float32
    #weight_init = tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype)
    W = _variable_with_weight_decay("W", weight_shape, stdev, wd)
    bias_init = tf.constant_initializer(value=0.1, dtype=dtype)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(input, kernel, stride):
    return tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')

def layer(input, weight_shape, bias_shape, stdev, wd):
    #dtype = tf.float16
    #weight_init = tf.truncated_normal_initializer(stddev=0.04, dtype=dtype)
    bias_init = tf.constant_initializer(value=0.1)
    W = _variable_with_weight_decay("W", weight_shape, stdev, wd)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b, name='last_layer')


def inference(x): #, keep_prob):

    #x = tf.reshape(x, shape=[-1, height, width, 3])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 64], [64], stdev=5e-2, wd=0.0)
        _activation_summary(conv_1)

    with tf.variable_scope('batch_norm'):
        norm1 = tf.nn.lrn(conv_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    with tf.variable_scope('max_pool1'):
        pool_1 = max_pool(norm1, kernel=2, stride=2)


    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], stdev=5e-2, wd=0.0)
        _activation_summary(conv_2)

    with tf.variable_scope('batch_norm'):
            norm2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    with tf.variable_scope('max_pool2'):
        pool_2 = max_pool(norm2,  kernel=2, stride=2)

    with tf.variable_scope("conv_3"):
        conv_3 = conv2d(pool_2, [5, 5, 64, 64], [64], stdev=5e-2, wd=0.0)
        _activation_summary(conv_3)

    with tf.variable_scope('batch_norm'):
            norm3 = tf.nn.lrn(conv_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm3')

    with tf.variable_scope('max_pool3'):
        pool_3 = max_pool(norm3,  kernel=2, stride=1)
    # with tf.variable_scope("conv_3"):
    #     conv_3 = conv2d(norm2, [5, 5, 64, 32], [32])

    # with tf.variable_scope('max_pool3',  kernel=3, stride=2):
    #     pool_3 = max_pool(conv_3)

    with tf.variable_scope("fc1"):
        # reshape = tf.reshape(pool_2, [batch_size, -1])
        # dim = reshape.get_shape()[1].value
        pool_2_flat = tf.reshape(pool_3, [-1, 25 * 25 * 64])
        fc_1 = layer(pool_2_flat, [25 * 25 * 64, 384], [384], stdev=0.04, wd=0.004)
        _activation_summary(fc_1)
    with tf.variable_scope("fc2"):
        fc_2 = layer(fc_1, [384, 192], [192], stdev=0.04, wd=0.004)
        # apply dropout
        #fc_2_drop = tf.nn.dropout(fc_2, keep_prob)
        _activation_summary(fc_2)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, nClass],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [nClass],
                                  tf.constant_initializer(0.0))
        output = tf.add(tf.matmul(fc_2, weights), biases, name=scope.name)
        _activation_summary(output)


    # with tf.variable_scope("output"):
    #     output = layer(fc_2, [192, nClass], [nClass], init_val=1/192.0, wd=0.0)
    return output

def loss(output, y):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(y, tf.int64)
  labels = tf.arg_max(labels, 1)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=output, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

# def loss(output, y):
#     xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
#     loss = tf.reduce_mean(xentropy)
#     return loss

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    num_batches_per_epoch = int(N_OF_SAMPL_TRAIN / batch_size)
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                               decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)



    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.minimize(cost, global_step=global_step)

    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(cost)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(cost)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op



def evaluate(output, y):
    logits = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate predictions.
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


x = tf.placeholder(tf.float32, [None, height, width, 3], name='placehold_x')
y = tf.placeholder(tf.int32, [None, nClass], name='placehold_y')

#keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout probability

output = inference(x) #, keep_prob)
tf.identity(output, name="inference")

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
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                train_precision = sess.run(eval_op, feed_dict={x: minibatch_x, y: minibatch_y})

                train_precision = sess.run(eval_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                # avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/batch_size
                # np.set_printoptions(precision=15)
                if not i % 10:
                    print('Epoch #: ', epoch, '  Batch #: ', i, 'train precision: ', train_precision, ' Global step:', sess.run(global_step))

            # Display logs per epoch step
            if epoch % display_step == 0:

                vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
                accuracy = sess.run(eval_op, feed_dict={x: vbatch_xs, y: vbatch_ys})
                print("Validation precision:", accuracy)

                temp1 = sess.run(output, feed_dict={x: vbatch_xs})
                prediction = sess.run(tf.nn.softmax(temp1))
                # np.set_printoptions(precision=10)
                print(vbatch_xs[0:5], '\n', vbatch_ys[0:5], '\n', prediction[0:5])

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
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


