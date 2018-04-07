import tensorflow as tf
import numpy as np
import timeit


def leaky_relu(input_, leakiness=0.2):
    assert leakiness <= 1
    return tf.maximum(input_, leakiness * input_)


def convt2d(input_, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, name="convt2d"):
    assert type(kernal) in [list, tuple, int]
    assert type(strides) in [list, tuple, int]
    assert type(padding) in [list, tuple, int, str]
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input_)[0]
    with tf.variable_scope(name):
        if type(padding) in [tuple, list, int]:
            if type(padding) == int:
                p_h = p_w = padding
            else:
                [p_h, p_w] = list(padding)
            pad_ = [0, p_h, p_w, 0]
            input_ = tf.pad(input_, [[p, p] for p in pad_], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=0.005))
        convt = tf.nn.conv2d_transpose(input_, w, output_shape=tf.stack(output_shape, axis=0), strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(convt, biases)
        if activate_fn:
            convt = activate_fn(convt)
        return convt


def generator(z_size, image_size, z, is_training=True):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        inputs = tf.reshape(z, [-1, 1, 1, z_size])
        convt1 = convt2d(inputs, (None, image_size // 16, image_size // 16, 512), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
        convt1 = leaky_relu(convt1)

        convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
        convt2 = leaky_relu(convt2)

        convt3 = convt2d(convt2, (None, image_size // 4, image_size // 4, 128), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
        convt3 = leaky_relu(convt3)

        convt4 = convt2d(convt3, (None, image_size // 2, image_size // 2, 64), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
        convt4 = leaky_relu(convt4)

        convt5 = convt2d(convt4, (None, image_size, image_size, 3), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt5")
        convt5 = tf.nn.tanh(convt5)

        return convt5


def langevin_generator_1(t, num_chain, z_size, sigma, delta, z_arg, y, generator):
    def cond(i, z):
        return tf.less(i, t)

    def body(i, z):
        noise = tf.random_normal(shape=[num_chain, z_size], name='noise')
        y_gen = generator(z)
        gen_loss = tf.reduce_mean(1.0 / (2 * sigma * sigma) * tf.square(y - y_gen), axis=0)
        grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
        z = z - 0.5 * delta * delta * (z + grad) + delta * noise
        return tf.add(i, 1), z

    with tf.name_scope("langevin_generator_1"):
        i = tf.constant(0)
        i, z = tf.while_loop(cond, body, [i, z_arg])
        return z


def langevin_generator_2(t, num_chain, z_size, sigma, delta, z, y, generator):
    Gy = generator(z)
    cost = (tf.nn.l2_loss(Gy - y) / (1*tf.pow(sigma,2))+tf.nn.l2_loss(z))/num_chain
    opt = tf.train.GradientDescentOptimizer(1)
    gv = opt.compute_gradients(cost,[z])
    def T(g):
        return delta*tf.random_normal([num_chain,z_size], stddev=1)+0.5*tf.square(delta)*g
    tgv = [(T(g),v) for (g,v) in gv]
    return opt.apply_gradients(tgv)


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    tf.set_random_seed(seed)

    t = 20
    num_chain = 64
    z_size = 100
    sigma = 1.0
    delta = 0.01
    image_size = 64

    z_val = np.random.randn(num_chain, z_size)
    y_val = np.random.randn(num_chain, image_size, image_size, 3)

    # langevin 1
    with tf.Graph().as_default() as graph:
        y = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32, name='obs')
        z = tf.placeholder(shape=[None, z_size], dtype=tf.float32, name='z')

        generator_z = lambda z : generator(z_size, image_size, z, is_training=True)
        langevin_1 = langevin_generator_1(t, num_chain, z_size, sigma, delta, z, y, generator_z)

        initialize = tf.global_variables_initializer()

        with tf.Session(graph=graph.finalize()) as sess:
            initialize.run()
            langevin_1_run = lambda: langevin_1.run(feed_dict={z: z_val, y: y_val})
            print(timeit.Timer(langevin_1_run).repeat(number=10))

    # langevin 2
    with tf.Graph().as_default() as graph:
        y = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32, name='obs')
        z = tf.placeholder(shape=[None, z_size], dtype=tf.float32, name='z')
        z2 = tf.Variable(z, name='z2', validate_shape=False)

        generator_z = lambda z : generator(z_size, image_size, z, is_training=True)
        langevin_2 = langevin_generator_2(t, num_chain, z_size, sigma, delta, z2, y, generator_z)

        initialize = tf.global_variables_initializer()

        with tf.Session(graph=graph.finalize()) as sess:
            initialize.run(feed_dict={z: z_val})
            def langevin_2_run():
                for i in range(t):
                    langevin_2.run(feed_dict={y: y_val})
            print(timeit.Timer(langevin_2_run).repeat(number=10))
