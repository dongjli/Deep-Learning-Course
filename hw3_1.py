import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_image = 28
n_labels = 10
eta = 0.3
nIter = 30000
batch_size = 500

x = tf.placeholder(tf.float32, [None, n_image*n_image])
y_ = tf.placeholder(tf.float32, [None, n_labels])

def weight_fun(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def bias_fun(shape):
    bias = tf.constant(0.0, shape=shape)
    return tf.Variable(bias)

############## Single Layer #################

W = weight_fun([n_image*n_image, n_labels])
b = bias_fun([n_labels])

output = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output)
)
train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Calculate accuracy 
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(nIter+1):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: x_batch, y_: y_batch})
    if i in [100, 1000, 5000, 10000, 20000, nIter]:
        train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Iteration %d, training accuracy %g testing accuracy: %g %%"
              %(i, train_accuracy*100, test_accuracy*100))


### Layer: [(50, ReLU), (50, ReLU), (10, Linear)] ###

x = tf.placeholder(tf.float32, [None, n_image*n_image])
y_ = tf.placeholder(tf.float32, [None, n_labels])

W1 = weight_fun([n_image*n_image, 50])
b1 = bias_fun([50])
output1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_fun([50, 50])
b2 = bias_fun([50])
output2 = tf.nn.relu(tf.matmul(output1, W2) + b2)

W3 = weight_fun([50, 10])
b3 = bias_fun([10])
output = tf.nn.softmax(tf.matmul(output2, W3) + b3)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output)
)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Calculate accuracy 
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(nIter+1):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: x_batch, y_: y_batch})
    if i in [100, 1000, 5000, 10000, 20000, nIter]:
        train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Iteration %d, training accuracy %g testing accuracy: %g %%"
              %(i, train_accuracy*100, test_accuracy*100))








