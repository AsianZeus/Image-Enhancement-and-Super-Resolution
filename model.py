import tensorflow as tf

def resnet(image):

    with tf.compat.v1.variable_scope("generator"):

        # Convolutional layer

        W1 = weight_variable([9, 9, 3, 64], name="W1");
        b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(image, W1) + b1)

        # Residual layer 1

        W2 = weight_variable([3, 3, 64, 64], name="W2");
        b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3");
        b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # Residual layer 2

        W4 = weight_variable([3, 3, 64, 64], name="W4");
        b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5");
        b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # Residual layer 3

        W6 = weight_variable([3, 3, 64, 64], name="W6");
        b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7");
        b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # Residual layer 4

        W8 = weight_variable([3, 3, 64, 64], name="W8");
        b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9");
        b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional layer

        W10 = weight_variable([3, 3, 64, 64], name="W10");
        b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        # Convolutional layer

        W11 = weight_variable([3, 3, 64, 64], name="W11");
        b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Output layer

        W12 = weight_variable([9, 9, 64, 3], name="W12");
        b12 = bias_variable([3], name="b12");
        prediction = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return prediction

def weight_variable(shape, name):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')

def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(x=net, axes=[1, 2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift
