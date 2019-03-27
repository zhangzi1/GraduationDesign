import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def conv2d(input, kernel_size, output_channel_num):
    kernel = tf.Variable(
        tf.truncated_normal([kernel_size, kernel_size, input.get_shape().as_list()[-1], output_channel_num],
                            stddev=0.05),
        name="kernel")
    bias = tf.Variable(tf.truncated_normal([output_channel_num], stddev=0.05), name="conv_bias")
    output = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], "SAME") + bias
    return output


def dense(input, output_size, function=None):
    weight = tf.Variable(tf.truncated_normal([input.get_shape().as_list()[-1], output_size], stddev=0.05),
                         name="weight")
    bias = tf.Variable(tf.truncated_normal([output_size], stddev=0.05), name="dense_bias")
    output = tf.matmul(input, weight) + bias
    if function is None:
        return output
    else:
        return function(output)


def minimize(optimizer, loss, vars, max_grad_norm):
    grads_and_vars = optimizer.compute_gradients(loss)
    new_grads_and_vars = []
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is not None and var in vars:
            new_grads_and_vars.append((tf.clip_by_norm(grad, max_grad_norm), var))
    return optimizer.apply_gradients(new_grads_and_vars)


features = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="input_batch")
labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

input_batch = tf.reshape(features, [-1, 28, 28, 1])

with tf.variable_scope("") as scp:
    outputs = conv2d(input_batch, 3, 64)
    outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    outputs = tf.nn.leaky_relu(outputs)

    outputs = conv2d(outputs, 3, 32)
    outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    outputs = tf.nn.leaky_relu(outputs)

    outputs = tf.reshape(outputs, [-1, 32 * 7 * 7])

    outputs = dense(outputs, 100, tf.nn.leaky_relu)
    outputs = dense(outputs, 10, tf.nn.softmax)
    vars = tf.contrib.framework.get_variables(scp)

cross_entropy = -tf.reduce_sum(labels * tf.log(outputs + 1e-10))
optimizer = tf.train.AdamOptimizer(0.001)
train_step = minimize(optimizer, cross_entropy, vars, 50)
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.scalar("Cross Entropy", cross_entropy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./graphs", sess.graph)

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(32)
    sess.run(train_step, {features: batch_x, labels: batch_y})
    if i % 10 == 0:
        batch_a, batch_b = mnist.test.next_batch(32)
        loss = sess.run(cross_entropy, feed_dict={features: batch_a, labels: batch_b})
        print(loss)
        summary = sess.run(merged_summary, feed_dict={cross_entropy: loss})
        writer.add_summary(summary, global_step=i)

print("Accuracy:",
      sess.run(train_accuracy,
               feed_dict={features: mnist.test.images, labels: mnist.test.labels}))
