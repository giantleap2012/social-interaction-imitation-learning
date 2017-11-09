import tensorflow as tf
import argparse
import os
import time
import pickle
import numpy as np

from data_input import DataLoader
from imitation_network import ImitationNetwork

# Input of the network
# Conv without max pooling + LSTM shape
# Output with directly MSE error -> This should be correct.
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--is_training', type=bool, default=True, help='is this training process?')
	parser.add_argument('--num_epoches', type=int, default=50, help='the number of epoches to train')
	parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
	parser.add_argument('--base_learning_rate', type=float, default=0.002, help='learning_rate')
	parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate for learning rate')
	parser.add_argument('--grad_clip', type=float, default=1., help='the grad clip')
	parser.add_argument('--data_folder', type=str, default='./data/3-person-simple-lab')
	parser.add_argument('--model_path', type=str, default='./model-ckpts')
	args = parser.parse_args()
	train(args)

def build_graph(args, global_step, lr, loss):
	tf.summary.scalar('learning_rate', lr)
	
	# Compute gradients.
	opt = tf.train.AdamOptimizer(lr)
	
	grads = tf.gradients(loss, tf.trainable_variables())
	grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')

	return train_op

def train(args):
	model_dir = args.model_path
	if model_dir is None:
		model_dir = './save'
	
	logfile_path = os.path.join(model_dir, 'logging.txt')
	logging_file = open(logfile_path, 'w')

	# Build computational graph
	with tf.Graph().as_default():
		model = ImitationNetwork(args)
		
		global_step = tf.Variable(0, name='global_step', trainable=False)

		images_placeholder = tf.placeholder(tf.float32, shape=[args.batch_size, 320, 320, 3])

		targets_placeholder = tf.placeholder(tf.float32, shape=[args.batch_size, 2])

		lr = tf.Variable(args.base_learning_rate, trainable=False, name="learning_rate")

		predictions = model.inference(images_placeholder)
		
		loss = model.loss(predictions, targets_placeholder)

		train_op = build_graph(args, global_step, lr, loss)

		sess = tf.Session()

		saver = tf.train.Saver(max_to_keep=100)

		ckpt = tf.train.get_checkpoint_state(model_dir)
		if ckpt is None:
			init = tf.global_variables_initializer()
			sess.run(init)
		else:
			print 'Loading Model from ' + ckpt.model_checkpoint_path
			saver.restore(sess, ckpt.model_checkpoint_path)


		training_data_loader = DataLoader(args)
		valid_data_loader = DataLoader(args, is_valid=True)

		best_epoch = -1
		best_loss_epoch = 0.0
		for curr_epoch in range(args.num_epoches):
			training_loss_epoch = 0.0
			valid_loss_epoch = 0.0

			############################################# Training process ######################################
			print 'Training epoch ' + str(curr_epoch + 1) + '........................'

			if curr_epoch % 10 == 0:
				sess.run(tf.assign(lr, args.base_learning_rate * (args.decay_rate ** curr_epoch / 10)))

			training_data_loader.reset_pointer()

			for step in range(training_data_loader.num_batches):
				start_time = time.time()

				images, targets = training_data_loader.next_batch()

				_, loss_batch = sess.run([train_op, loss], feed_dict={
						images_placeholder: images,
						targets_placeholder: targets})

				end_time = time.time()
				training_loss_epoch += loss_batch
				print("Training {}/{} (epoch {}), train_loss = {:.8f}, time/batch = {:.3f}"
					.format(
						step + 1,
						training_data_loader.num_batches,
						curr_epoch + 1,
						loss_batch, end_time - start_time))

			print 'Epoch ' + str(curr_epoch + 1) + ' training is done! Saving model...'
			checkpoint_path = os.path.join(model_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=global_step)

			############################################# Validating process ######################################
			print 'Validating epoch ' + str(curr_epoch + 1) + '...........................'
			valid_data_loader.reset_pointer()
			
			for step in range(valid_data_loader.num_batches):
				start_time = time.time()

				images, targets = valid_data_loader.next_batch()

				loss_batch = sess.run(loss, feed_dict={
						images_placeholder: images,
						targets_placeholder: targets}
						)

				end_time = time.time()
				valid_loss_epoch += loss_batch
				print("Validating {}/{} (epoch {}), valid_loss = {:.8f}, time/batch = {:.3f}"
					.format(
						step + 1,
						valid_data_loader.num_batches,
						curr_epoch + 1, 
						loss_batch, end_time - start_time))

			# Update best valid epoch
			if best_epoch == -1 or best_loss_epoch > valid_loss_epoch:
				best_epoch = curr_epoch + 1
				best_loss_epoch = valid_loss_epoch

			logging_file.write('epoch ' + str(curr_epoch + 1) + '\n')
			logging_file.write(str(curr_epoch + 1) + ',' + str(training_loss_epoch) + '\n')
			logging_file.write(str(curr_epoch + 1) + ',' + str(valid_loss_epoch) + '\n')
			logging_file.write(str(best_epoch) + ',' + str(best_loss_epoch) + '\n')

		logging_file.close()

if __name__ == '__main__':
    main()
