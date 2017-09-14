from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

import get_data as gd


class ConvNet:
    def __init__(self, total_samples=52368, samples=300, split=0.8,
                 image_width=128, image_height=44, channels=1, classes=2):
        # HyperParameters
        self.TOTAL_SAMPLES = total_samples
        self.SAMPLES = samples
        self.SPLIT = split
        self.TRAIN_SIZE = int(samples * split)
        self.IMAGE_HEIGHT = image_height
        self.IMAGE_WIDTH = image_width
        self.CHANNELS = channels  # Number of color channels (RGB = 3)
        self.CLASSES = classes

        # TensorFlow class for performing high-level model training, evaluation, and inference
        self.classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn, model_dir=args.model_dir)

    def generate_indices(self, start, end, seed=0):
        indices = np.arange(self.TOTAL_SAMPLES)
        np.random.seed(seed)
        np.random.shuffle(indices)
        return indices[start:end]

    def set_training_data(self):
        train_indices = self.generate_indices(0, self.TRAIN_SIZE)
        self.train_data = gd.get_melspectrograms(train_indices)
        self.train_labels = np.asarray(gd.get_labels(train_indices))

        if args.verbose:
            unique, counts = np.unique(NN.train_labels, return_counts=True)
            print('Train samples: {}, classes: {}'.format(
                NN.train_labels.shape[0], dict(zip(unique, counts))))

    def set_eval_data(self):
        eval_indices = self.generate_indices(self.TRAIN_SIZE, self.SAMPLES)
        self.eval_data = gd.get_melspectrograms(eval_indices)
        self.eval_labels = np.asarray(gd.get_labels(eval_indices))

        if args.verbose:
            unique, counts = np.unique(NN.eval_labels, return_counts=True)
            print('Eval samples: {}, classes: {}'.format(
                NN.eval_labels.shape[0], dict(zip(unique, counts))))

    def cnn_model_fn(self, features, labels, mode):
        # Input Layer
        # Reshape to 4D tensor: [batch_size, width, height, channels]
        # Batch size: Size of the subset of examples to use when performing
        # gradient descent during training. -1 specifies that this dimension
        # should be dynamically computed based on the number of input values in features
        input_layer = tf.reshape(features['data'], [-1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.CHANNELS])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

        # Pooling layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten our feature map to [batch_size, features]
        pool2_flat = tf.reshape(pool2, [-1, 11 * 32 * 64])

        # Dense Layer
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # Also apply dropout regularization to our dense layer to improve the results
        # (probability of 'rate' that any given element will be dropped during training)
        dropout = tf.layers.dropout(
            inputs=dense, rate=args.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer. Raw values of our predictions
        # One neuron for each target class
        # Output logits shape [batch_size, CLASSES]
        logits = tf.layers.dense(inputs=dropout, units=self.CLASSES)

        # EstimatorSpec arguments generation
        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

        loss = None
        train_op = None
        eval_metric_ops = None

        if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.CLASSES)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    def train(self, data, labels):
        # Set up logging for predictions
        # Log the values in the 'Softmax' tensor with label 'probabilities'
        tensors_to_log = {'probabilities': 'softmax_tensor'}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=75)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'data': data},
            y=labels,
            batch_size=100,
            num_epochs=None,
            shuffle=False)
        self.classifier.train(
            input_fn=train_input_fn,
            steps=args.max_steps,
            hooks=[logging_hook])

    def evaluate(self, data, labels):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'data': data},
            y=labels,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        return eval_results

    def make_verbose(self, verbose):
        if verbose:
            tf.logging.set_verbosity(tf.logging.INFO)
        else:
            tf.logging.set_verbosity(tf.logging.ERROR)
            args.verbose = verbose


def get_error_list(indices):
    import pandas
    csv = pandas.read_csv('./labels.csv', header=0)
    errors = []

    for idx in indices:
        eval_results = NN.evaluate(gd.get_melspectrograms([idx]),
                                   np.asarray(gd.get_labels([idx])))
        if eval_results['accuracy'] == 0.0:
            errors.append(csv['path'][idx])

    return errors


def generate_playlist(files):
    with open('errors.m3u', 'w') as playlist:
        for file in files:
            playlist.write(file + '\n')


def main(_):
    NN.make_verbose(True)
    if args.should_train:
        NN.set_training_data()
        # Train the model
        NN.train(NN.train_data, NN.train_labels)

    if args.should_test:
        NN.set_eval_data()
        NN.make_verbose(False)
        # Evaluate the model and print results
        print('Result: ', NN.evaluate(NN.eval_data, NN.eval_labels))
        generate_playlist(get_error_list(NN.generate_indices(NN.TRAIN_SIZE, NN.SAMPLES)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', action='store_true', dest='should_train')
    parser.add_argument('-te', action='store_true', dest='should_test')
    parser.add_argument('-v', action='store_true', dest='verbose')
    parser.add_argument('--samples', type=int, default=300, dest='samples', help='Number of samples to use')
    parser.add_argument('--max_steps', type=int, default=100, dest='max_steps', help='Number of steps to run trainer')
    parser.add_argument('--learning_rate', type=float, default=0.001, dest='learning_rate', help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.4, dest='dropout', help='Keep probability for training dropout')
    parser.add_argument('--model_dir',type=str, default='./models/model', help='Directory for storing model data')
    #parser.add_argument('--logs_dir',type=str, default='/tmp/test_logs', help='Directory for storing logs data')
    args = parser.parse_args()
    NN = ConvNet(samples = args.samples)
    tf.app.run()
