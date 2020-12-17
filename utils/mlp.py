import tensorflow as tf
from pdb import set_trace as st

def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input
