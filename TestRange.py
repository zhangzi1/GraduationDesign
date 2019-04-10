import time

import tensorflow as tf
import tensorflow.models.tutorials.image.cifar10.cifar10_input as cifar10_input


def conv2d(input, kernel_size, output_channel_num, activation_function=None):
    kernel = tf.Variable(
        tf.truncated_normal([kernel_size, kernel_size, input.get_shape().as_list()[-1], output_channel_num],
                            stddev=0.05),
        name="kernel")
    bias = tf.Variable(tf.truncated_normal([output_channel_num], stddev=0.05), name="conv_bias")
    output = tf.add(tf.nn.conv2d(input, kernel, [1, 1, 1, 1], "SAME"), bias)
    if activation_function is None:
        return output
    else:
        return activation_function(output)


def dense(input, output_size, keep_prob, activation_function=None):
    weight = tf.Variable(tf.truncated_normal([input.get_shape().as_list()[-1], output_size], stddev=0.05),
                         name="weight")
    bias = tf.Variable(tf.truncated_normal([output_size], stddev=0.05), name="dense_bias")
    output = tf.add(tf.matmul(input, weight), bias)
    if activation_function is None:
        pass
    else:
        output = activation_function(output)
    return tf.nn.dropout(output, keep_prob)


def minimize(optimizer, loss, vars, max_grad_norm):
    grads_and_vars = optimizer.compute_gradients(loss)
    new_grads_and_vars = []
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is not None and var in vars:
            new_grads_and_vars.append((tf.clip_by_norm(grad, max_grad_norm), var))
    return optimizer.apply_gradients(new_grads_and_vars)


def num2onehot(label_batch):
    output_batch = []
    for i in range(len(label_batch)):
        zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zeros[label_batch[i]] = 1
        output_batch.append(zeros)
    return output_batch


if __name__ == '__main__':
    # 占位符
    features = tf.placeholder(tf.float32, shape=[None, 24, 24, 3], name="input_batch")
    labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # 网络
    with tf.variable_scope("") as scp:
        outputs = conv2d(features, 3, 32, tf.nn.leaky_relu)
        outputs = conv2d(outputs, 3, 32, tf.nn.leaky_relu)
        outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        outputs = conv2d(outputs, 3, 64, tf.nn.leaky_relu)
        outputs = conv2d(outputs, 3, 64, tf.nn.leaky_relu)
        outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        outputs = conv2d(outputs, 3, 128, tf.nn.leaky_relu)
        outputs = conv2d(outputs, 3, 128, tf.nn.leaky_relu)
        outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        outputs = tf.reshape(outputs, [-1, 128 * 3 * 3])
        outputs = dense(outputs, 100, keep_prob, tf.nn.leaky_relu)
        outputs = dense(outputs, 10, keep_prob, tf.nn.softmax)
        vars = tf.contrib.framework.get_variables(scp)

    # 损失函数、优化器、梯度截取
    cross_entropy = -tf.reduce_sum(labels * tf.log(outputs + 1e-10))
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train_step = minimize(optimizer, cross_entropy, vars, 50)

    # 正确率
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 标量图
    tf.summary.scalar("Cross Entropy", cross_entropy)
    tf.summary.scalar("Accuracy", train_accuracy)
    merged_summary = tf.summary.merge_all()

    # 会话
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 写文件
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    writer_train = tf.summary.FileWriter("./logss/" + localtime + "/train/", sess.graph)
    writer_test = tf.summary.FileWriter("./logss/" + localtime + "/test/")

    # 提取数据
    train_batch_x, train_batch_y = cifar10_input.distorted_inputs("./cifar-10-batches-bin/", 32)
    test_batch_x, test_batch_y = cifar10_input.inputs(True, "./cifar-10-batches-bin/", 32)
    tf.train.start_queue_runners()
