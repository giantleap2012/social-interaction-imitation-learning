import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import BasicLSTMCell
import tensorflow.contrib.slim as slim


class ImitationNetwork():
	def __init__(self, args):
		self.args = args
		self.batch_size = args.batch_size
		
		with tf.name_scope("LSTM_1"):
			self.lstm_1_size = 200
			self.cell_1 = BasicLSTMCell(self.lstm_1_size, state_is_tuple=False)
			self.cell_state_1 = tf.zeros([1, self.cell_1.state_size])

		with tf.name_scope("LSTM_2"):
			self.lstm_2_size = 100
			self.cell_2 = BasicLSTMCell(self.lstm_2_size, state_is_tuple=False)
			self.cell_state_2 = tf.zeros([1, self.cell_2.state_size])

	def cnn_network(self, input_batch_images):
		network = input_batch_images
		
		with tf.variable_scope("conv1"):
			# First conv layer
			conv1_kernel = tf.get_variable(name="kernel", shape=[3, 3, 3, 10], initializer=tf.contrib.layers.xavier_initializer())
			network = tf.nn.conv2d(
				network, conv1_kernel, [1, 1, 1, 1], "SAME", name="conv")
			network = slim.batch_norm(network, scope="conv1_bn")
			network = tf.nn.relu6(network)
			network = tf.nn.max_pool(network, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID", name="pooling")

		with tf.variable_scope("conv2"):
			# Second conv layer
			conv2_kernel = tf.get_variable(name="kernel", shape=[3, 3, 10, 10], initializer=tf.contrib.layers.xavier_initializer())
			network = tf.nn.conv2d(
				network, conv2_kernel, [1, 1, 1, 1], "SAME", name="conv")
			network = slim.batch_norm(network, scope="conv2_bn")
			network = tf.nn.relu6(network)
			network = tf.nn.max_pool(network, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID", name="pooling")

		with tf.variable_scope("conv3"):
			# Third conv layer
			conv3_kernel = tf.get_variable(name="kernel", shape=[3, 3, 10, 10], initializer=tf.contrib.layers.xavier_initializer())
			network = tf.nn.conv2d(
				network, conv3_kernel, [1, 1, 1, 1], "SAME", name="conv")
			network = slim.batch_norm(network, scope="conv3_bn")
			network = tf.nn.relu6(network)
			network = tf.nn.max_pool(network, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID", name="pooling")

		return network

	def inference(self, input_batch_images):
		batch_features = self.cnn_network(input_batch_images)

		with tf.variable_scope("lstm") as scope:
			for i in range(self.args.batch_size):
				curr_image_features = batch_features[i, :, :, :]
				curr_image_features = tf.reshape(curr_image_features, [1, -1])

				with tf.variable_scope("cell_1") as scope1:
					if i > 0:
						scope1.reuse_variables()
					output_1, self.cell_state_1 = self.cell_1(curr_image_features, self.cell_state_1)

				with tf.variable_scope("cell_2") as scope2:
					if i > 0:
						scope2.reuse_variables()
					output_2, self.cell_state_2 = self.cell_2(output_1, self.cell_state_2)

				prediction = tf.contrib.layers.fully_connected(output_2, 2)
				if i == 0:
					predictions = prediction
				else:
					predictions = tf.concat([predictions, prediction], axis=0)

		return predictions

	def loss(self, predictions, targets):
		with tf.variable_scope("loss") as scope:
			# linear_speed_weight = 1.5
			# angular_speed_weight = 1.0
			
			# linear_targets = tf.multiply(tf.divide(targets[:, 0], 6.92), linear_speed_weight)
			# angular_targets = tf.multiply(tf.divide(targets[:, 1], 0.26), angular_speed_weight)
			
			# targets = tf.concat([tf.reshape(linear_targets, [-1, 1]), tf.reshape(angular_targets, [-1, 1])], axis=1)

			mse = tf.reduce_sum(tf.square(targets - predictions))
		return mse

