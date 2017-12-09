import itertools
import sys
from collections import deque
from random import sample, shuffle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
from numpy.random import randint
from numpy.random import uniform
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

import dnc

# TODO:
# Add input data in answer phase (based structured prediction section of the paper)
# Add loss function
# Add proper training loop
# Add curriculum learning

GRAPH_SIZE = 3
MAX_NODE_NEIGHBOURS = 3
PLANNING_STEPS = 10

DEBUG = False
SHOW = False

# These constants cannot be changed without changing the code that relies on them
INPUT_VECTOR_SIZE = 92
TRANSITION_CHANNEL_INDEX = 90
ANSWER_CHANNEL_INDEX = 91
TARGET_VECTOR_SIZE = 90

NUM_DIGITS = 9

BATCH_SIZE = 16

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 10,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 100,
                        "Checkpointing step interval.")


def generate_graph(graph_size, max_neighbours):
    # "For all graph tasks, the graphs used to train the networks were generated
    # by uniformly sampling a set of two-dimensional points from a unit square,
    # each point corresponding to a node in the graph."
    x_coords = uniform(size=graph_size)
    y_coords = uniform(size=graph_size)

    # "The numerical labels for the nodes were chosen uniformly from the
    # range [0, 999]"
    labels = sample(range(1000), graph_size)

    # "For each node, the K nearest neighbours in the square were used as the K
    # outbound connections, with K independently sampled from a uniform range
    # for each node."
    neighbour_counts = randint(low=2, high=max_neighbours + 1, size=graph_size)

    # "For a graph with N nodes, N unique numbers in the range [0, 999] were
    # initially drawn."
    edge_labels = sample(range(1000), graph_size)

    node_list = [node_data for node_data in zip(x_coords, y_coords,
                                                labels, neighbour_counts)]

    def distance(node1, node2):
        return ((node1[0] - node2[0]) ** 2) + \
               ((node1[1] - node2[1]) ** 2)

    graph = nx.DiGraph()

    for node_data in node_list:

        x_pos, y_pos, label, neighbour_count = node_data

        graph.add_node(label, attr_dict={'position': (x_pos, y_pos)})

        # Sort by distance to other nodes
        nearest = sorted([x for x in node_list if x[2] != label],
                         key=lambda target: distance(node_data, target))

        # "For a graph with N nodes, N unique numbers in the range [0, 999] were
        # initially drawn. Then, the outbound edge labels for each node were
        # chosen at random from those N numbers."
        shuffled_edge_labels = list(edge_labels)
        shuffle(shuffled_edge_labels)
        edge_labels_shuffled = deque(iterable=shuffled_edge_labels)

        for i in range(min(neighbour_count, graph_size - 1)):
            neighbour = nearest[i]
            graph.add_edge(label, neighbour[2],
                           attr_dict={'label': edge_labels_shuffled.pop()})

    return graph


def show_graph(graph):
    nx.draw_networkx(graph, pos={node: data_dict['position']
                                 for node, data_dict in graph.nodes(data=True)},
                     font_size=9, node_size=400, alpha=0.9,
                     node_color='#B0E0E6')
    plt.show()


def get_positions(label, offset):
    return tuple([offset + (i * 10) + int(str(label).zfill(3)[i])
                  for i in range(3)])


def encode_one_hot(input_vector, node1, node2, edge):
    positions_to_set = []
    positions_to_set.extend(get_positions(node1, 0))
    if edge is not None:
        positions_to_set.extend(get_positions(edge, 30))
    positions_to_set.extend(get_positions(node2, 60))
    for position in positions_to_set:
        input_vector[position] = 1
    return input_vector


def get_edge_vector(node1, node2, edge):
    # "Each vector encoded a triple consisting of a source label, an edge
    # label and a destination label. All labels were represented as
    # numbers between 0 and 999, with each digit represented as a
    # 10-way one-hot encoding."
    input_vector = np.zeros(shape=INPUT_VECTOR_SIZE)
    return encode_one_hot(input_vector, node1, node2, edge)


def get_transition_vector():
    transition_vector = np.zeros(shape=INPUT_VECTOR_SIZE)
    # Set the transition channel to 1
    transition_vector[TRANSITION_CHANNEL_INDEX] = 1
    return transition_vector


def get_planning_vector():
    return np.zeros(shape=INPUT_VECTOR_SIZE)


def get_answer_vector():
    transition_vector = np.zeros(shape=INPUT_VECTOR_SIZE)
    # Set the answer channel to 1
    transition_vector[ANSWER_CHANNEL_INDEX] = 1
    return transition_vector


def get_empty_target_vector():
    return np.zeros(shape=TARGET_VECTOR_SIZE)


def get_termination_pattern():
    termination_vector = get_empty_target_vector()
    for i in range(len(termination_vector)):
        if i % 2 == 0:
            termination_vector[i] = 1
    return termination_vector


def get_training_data(graph):
    # The training_data returned consists of a vector for each time step of
    # training.
    #
    # Training consists of a description phase, in which every edge
    # in the graph is presented along with its associated nodes, in
    # random order, followed by a series of question and answer
    # phases (how many is determined by the questions parameter).
    # Each question and answer phase consists of a single time step
    # for the question, then 10 time steps to allow for planning, then
    # some number of time steps during which the answer is expected.
    #
    # Two additional channels are present in the training data. From the paper:
    # "The input vectors had additional binary channels alongside the triples to
    # indicate when transitions between the different phases occurred, and when
    # a prediction was required of the network (this channel remained active
    # throughout the answer phase). In total, the input vectors were size 92
    # and the target vectors were size 90."
    #
    # So the size of the training vector is always 92: 90 for one-hot encoding
    # of the two node labels and the edge label, and 2 for the two additional
    # channels.

    # Generate a vector for each edge of the graph, in random order.
    description_phase = []
    for edge in graph.edges(data=True):
        node1, node2, data_dict = edge
        description_phase.append(get_edge_vector(node1, node2,
                                                 data_dict['label']))
    shuffle(description_phase)

    # Generate as many empty target vectors as description vectors, so that the lengths match
    target_description_phase = [get_empty_target_vector()] * len(description_phase)

    # Generate a question and answer for every combination of nodes in the graph
    # where there is a path between them
    q_a_phases = []
    target_q_a_phases = []
    for start_node in graph.nodes():
        for end_node in graph.nodes():

            if start_node != end_node and nx.has_path(graph, start_node, end_node):

                path = nx.shortest_path(graph, start_node, end_node)

                q_a_phases.append(get_transition_vector())
                target_q_a_phases.append(get_empty_target_vector())

                # This is the question phase
                q_a_phases.append(get_edge_vector(start_node, end_node, None))
                target_q_a_phases.append(get_empty_target_vector())

                q_a_phases.append(get_transition_vector())
                target_q_a_phases.append(get_empty_target_vector())

                # This is the planning phase
                for i in range(PLANNING_STEPS):
                    q_a_phases.append(get_planning_vector())
                    target_q_a_phases.append(get_empty_target_vector())

                q_a_phases.append(get_transition_vector())
                target_q_a_phases.append(get_empty_target_vector())

                edge_dict = {(node1, node2): data_dict
                             for node1, node2, data_dict in graph.edges(data=True)}

                # This is the answer phase. We need as many answer time steps as there are
                # nodes in the shortest path.
                previous_node = path[0]
                for i in range(1, len(path)):
                    next_node = path[1]
                    edge_data = edge_dict[(previous_node, next_node)]
                    edge_label = edge_data['label']
                    q_a_phases.append(get_answer_vector())
                    target_q_a_phases.append(encode_one_hot(get_empty_target_vector(),
                                                            previous_node, next_node, edge_label))
                    previous_node = next_node

                # Append a final answer vector for the termination pattern:
                # "the network indicated that it had completed an answer by
                # outputting a specially reserved termination pattern"
                q_a_phases.append(get_answer_vector())
                target_q_a_phases.append(get_termination_pattern())

    observation_data = np.asarray(description_phase + q_a_phases)
    target_data = np.asarray(target_description_phase + target_q_a_phases)

    return observation_data, target_data


def print_edges(graph):
    print('GRAPH EDGES:')
    for from_node, to_node, node_data in graph.edges(data=True):
        print('From {0} to {1} with: {2}'
              .format(from_node, to_node, node_data))


def generate_data(graph_size, batch_size):
    observation_batch = []
    target_batch = []
    for i in range(batch_size):
        debug('Generating graph for batch index {0}'.format(i))
        graph = generate_graph(graph_size, MAX_NODE_NEIGHBOURS)
        if DEBUG:
            print_edges(graph)
        if SHOW:
            show_graph(graph)
        observation_data, target_data = get_training_data(graph)
        observation_batch.append(observation_data)
        target_batch.append(target_data)
    num_time_steps = len(observation_batch[0])
    debug('Num time steps: {0}'.format(num_time_steps))
    for observation in observation_batch:
        assert len(observation) == num_time_steps
    return np.asarray(observation_batch), np.asarray(target_batch), num_time_steps


def run_model(training_data, output_size):
    """Based on the example in train.py"""

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        "hidden_size": FLAGS.hidden_size,
    }
    clip_value = FLAGS.clip_value

    dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
    initial_state = dnc_core.initial_state(batch_size=BATCH_SIZE)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=dnc_core,
        inputs=training_data,
        time_major=False,
        initial_state=initial_state)

    return output_sequence


def get_edge_label(time_step_data):
    edge_label = ['(']
    for label_index in range(3):
        for digit_index in range(3):
            one_hot_digit_start = (label_index * 30) + (digit_index * 10)
            one_hot_digit = time_step_data[one_hot_digit_start:one_hot_digit_start + 10]
            digit = sum(a * b for a, b in zip(one_hot_digit, range(10)))
            edge_label.append(str(int(digit)))
        if label_index < 2:
            edge_label.append(',')
    edge_label.append(')')
    return edge_label


def log_training_data(observation_data, target_data):

    for i, batch_data in enumerate(observation_data):

        print('BATCH {0}:'.format(i))
        print()

        phase = 'DESCRIPTION'
        transition = False

        for j, time_step_data in enumerate(batch_data):

            if time_step_data[TRANSITION_CHANNEL_INDEX] == 1:
                transition = True
                print('TIME STEP {0} (TRANSITION):'.format(j, phase))
            else:
                print('TIME STEP {0} ({1} phase):'.format(j, phase))

            print('RAW DATA:')
            print(time_step_data)
            print()

            edge_label = get_edge_label(time_step_data)
            print('EDGE DATA:')
            print(''.join(edge_label))
            print()

            print('PHASE TRANSITION CHANNEL: {0}'.format(time_step_data[TRANSITION_CHANNEL_INDEX]))
            print('ANSWER CHANNEL: {0}'.format(time_step_data[ANSWER_CHANNEL_INDEX]))
            print()

            if transition:
                transition = False
                if phase == 'DESCRIPTION':
                    phase = 'QUESTION'
                elif phase == 'QUESTION':
                    phase = 'PLANNING'
                elif phase == 'PLANNING':
                    phase = 'ANSWER'
                elif phase == 'ANSWER':
                    phase = 'QUESTION'
                else:
                    raise RuntimeError('Invalid phase: {0}'.format(phase))

            target_time_step = target_data[i][j]

            if np.sum(target_time_step) == 0:
                print('TARGET: EMPTY')
            else:
                print('TARGET: NON-EMPTY')
                print('TARGET RAW:')
                print(target_time_step)
                if np.array_equal(target_time_step, get_termination_pattern()):
                    print('TERMINATION PATTERN!')
                else:
                    edge_label = get_edge_label(target_time_step)
                    print('TARGET EDGE:')
                    print(''.join(edge_label))

            print('')
            print('-----------------------------------------------------------------------------')
            print('')


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def assert_shape(array_or_tensor, shape):

    if len(array_or_tensor.shape) != len(shape):
        raise ValueError('Rank of tensor does not match expected rank. Tensor '
                         'is rank {0} with shape {1}, but expected rank {2} with '
                         'shape {3}.'.format(len(array_or_tensor.shape), array_or_tensor.shape,
                                             len(shape), shape))

    for i, size in enumerate(shape):
        if size is not None:
            if array_or_tensor.shape[i] != size:
                raise ValueError('Size of array at axis {0} does not match. Tensor '
                                 'has shape {1}, but expected shape is shape {2}.'
                                 .format(i, array_or_tensor.shape, shape))

    if DEBUG:
        print('Array {0} has shape of {1}'.format(array_or_tensor, shape))


def debug(debug_string):
    if DEBUG:
        print(debug_string)


def digits_tensors(tensor):
    digits = tf.split(tensor, num_or_size_splits=NUM_DIGITS, axis=2)
    debug('Digits tensor: {0}'.format(digits))
    return digits


def get_ce_loss(logits, labels):
    ce_loss = softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    debug('CE loss tensor: {0}'.format(ce_loss))
    return ce_loss


def calculate_loss(output_tensor, target_tensor, mask_tensor, batch_size, num_time_steps):

    # TODO: It would likely be more efficient to perform masking before calculating the loss
    # by filtering the output and target tensors

    assert_shape(output_tensor, (batch_size, num_time_steps, TARGET_VECTOR_SIZE))
    assert_shape(target_tensor, (batch_size, num_time_steps, TARGET_VECTOR_SIZE))
    assert_shape(mask_tensor, (batch_size, num_time_steps))

    # These are actually tuples of tensors, of size NUM_DIGITS
    output_digits_tensors = digits_tensors(output_tensor)
    target_digits_tensors = digits_tensors(target_tensor)

    # Calculate the cross-entropy loss
    ce_loss = get_ce_loss(logits=output_digits_tensors, labels=target_digits_tensors)

    # The CE loss is calculated separately for each digit, so we now have a tensor
    # where the first dimension represents the digit, the second represents the batch size,
    # and the third represents the number of time steps. A loss value is present for each
    # time step.
    assert_shape(ce_loss, (NUM_DIGITS, batch_size, num_time_steps))

    # We sum the loss across all digits now so we're left with a tensor which represents all
    # of the time steps for each batch
    loss_across_digits = tf.reduce_sum(ce_loss, axis=0)
    debug('Loss across digits: {0}'.format(loss_across_digits))
    assert_shape(loss_across_digits, (batch_size, num_time_steps))

    # We can now apply the mask, which is of shape batch_size, num_time_steps
    masked_loss = loss_across_digits * mask_tensor
    debug('Masked loss: {0}'.format(masked_loss))
    assert_shape(masked_loss, (batch_size, num_time_steps))

    # We can now compute the total loss for each item in the batch, simply by summing across
    # all time steps
    batch_loss = tf.reduce_sum(masked_loss, axis=1)
    debug('Batch loss: {0}'.format(batch_loss))
    assert_shape(batch_loss, (batch_size,))

    mean_loss = tf.reduce_sum(batch_loss) / batch_size
    debug('Mean loss: {0}'.format(mean_loss))
    assert_shape(mean_loss, ())

    return mean_loss


def create_mask_data(observation_data):

    mask_data = []

    for batch_data in observation_data:
        mask_for_batch = []
        for time_step_data in batch_data:
            if time_step_data[ANSWER_CHANNEL_INDEX] == 1:
                mask_for_batch.append(1)
            else:
                mask_for_batch.append(0)
        mask_data.append(mask_for_batch)

    return np.asarray(mask_data)


def train(num_training_iterations, report_interval):
    """Trains the DNC and periodically reports the loss."""

    # Generate an initial batch of data, just so we know the shape of the data
    # for creating placeholders
    observation_data, target_data, num_time_steps = generate_data(graph_size=GRAPH_SIZE,
                                                                  batch_size=BATCH_SIZE)

    assert_shape(observation_data, (BATCH_SIZE, num_time_steps, INPUT_VECTOR_SIZE))
    assert_shape(target_data, (BATCH_SIZE, num_time_steps, TARGET_VECTOR_SIZE))

    if DEBUG:
        log_training_data(observation_data, target_data)

    mask_tensor = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, num_time_steps))

    observation_tensor = tf.placeholder(dtype=tf.float32, shape=observation_data.shape)
    target_tensor = tf.placeholder(dtype=tf.float32, shape=target_data.shape)

    output_logits = run_model(observation_tensor, output_size=TARGET_VECTOR_SIZE)

    # Used for visualization.
    output = tf.round(
        tf.expand_dims(mask_tensor, -1) * tf.sigmoid(output_logits))

    train_loss = calculate_loss(output_logits, target_tensor, mask_tensor,
                                batch_size=BATCH_SIZE, num_time_steps=num_time_steps)

    # Set up optimizer with global norm clipping.
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver)
        ]
    else:
        hooks = []

    # Train.
    with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
        start_iteration = sess.run(global_step)
        total_loss = 0

        for train_iteration in range(start_iteration, num_training_iterations):

            observation_data, target_data, num_time_steps = generate_data(graph_size=GRAPH_SIZE,
                                                                          batch_size=BATCH_SIZE)

            mask_data = create_mask_data(observation_data)

            assert_shape(observation_data, (BATCH_SIZE, num_time_steps, INPUT_VECTOR_SIZE))
            assert_shape(target_data, (BATCH_SIZE, num_time_steps, TARGET_VECTOR_SIZE))
            assert_shape(mask_data, (BATCH_SIZE, num_time_steps))

            _, loss = sess.run([train_step, train_loss], feed_dict={
                observation_tensor: observation_data,
                target_tensor: target_data,
                mask_tensor: mask_data
            })
            total_loss += loss

            if (train_iteration + 1) % report_interval == 0:
                tf.logging.info("%d: Avg training loss: %f",
                                train_iteration, total_loss / report_interval)
                total_loss = 0


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
    tf.app.run()
