import os
import numpy as np
import pickle
import cv2

class DataLoader():

	def __init__(self, args=None, preprocess=False, is_valid=False):
		self.data_folder = args.data_folder
		self.batch_size = args.batch_size

		# self.data_folder = '/cvgl2/u/junweiy/jackrabbot/social_imitation_learning/data/3-person-simple-lab'
		# self.batch_size = 64
		self.is_valid = is_valid

		# Preprocess image files and target file into pickle file
		if preprocess:
			self.preprocess()

		self.pointer = 0
		self.load_preprocessed()

	def preprocess(self):
		images = []
		image_filenames = []

		for filename in os.listdir(self.data_folder):			
			file_path = os.path.join(self.data_folder, filename)

			if filename.startswith('.'):
				continue
			
			if filename.endswith('jpg'):
				image_filenames.append(int(filename[:6]))
			
			if filename.endswith("csv"):
				targets = np.genfromtxt(file_path, delimiter=',')


		image_filenames.sort()
		for image_filename in image_filenames:
			image_name = str(image_filename).zfill(6) + '.jpg'
			print 'processing......', image_name
			file_path = os.path.join(self.data_folder, image_name)
			images.append(cv2.imread(file_path))

		index_split = len(targets) - len(targets) / 5
		training_images = images[:index_split]
		testing_images = images[index_split:]
		training_targets = targets[:index_split]
		testing_targets = targets[index_split:]

		train_data_file = os.path.join(self.data_folder, 'training.cpkl')
		f = open(train_data_file, "wb")
		pickle.dump((training_images, training_targets), f, protocol=2)
		f.close()

		test_data_file = os.path.join(self.data_folder, 'test.cpkl')
		f = open(test_data_file, 'wb')
		pickle.dump((testing_images, testing_targets), f, protocol=2)
		f.close()


	def load_preprocessed(self):
		if self.is_valid:
			file = os.path.join(self.data_folder, 'test.cpkl')
		else:
			file = os.path.join(self.data_folder, 'training.cpkl')

		f = open(file, 'rb')
		data = pickle.load(f)
		f.close()

		self.images, self.targets = data[0], data[1]
		print 'length of images: ', len(self.images)
		print 'length of targets: ', len(self.targets)

		abs_targets = np.absolute(self.targets)
		print '-----------------------------------------'
		print 'min: ', np.min(abs_targets[:, 0])
		print '1/4 percentile: ', np.percentile(abs_targets[:, 0], 25)
		print '2/4 percentile: ', np.percentile(abs_targets[:, 0], 50)
		print '3/4 percentile: ', np.percentile(abs_targets[:, 0], 75)
		print 'max: ', np.max(abs_targets[:, 0])
		print 'mean: ', np.mean(abs_targets[:, 0])

		print '-----------------------------------------'
		print 'min: ', np.min(abs_targets[:, 1])
		print '1/4 percentile: ', np.percentile(abs_targets[:, 1], 25)
		print '2/4 percentile: ', np.percentile(abs_targets[:, 1], 50)
		print '3/4 percentile: ', np.percentile(abs_targets[:, 1], 75)
		print 'max: ', np.max(abs_targets[:, 1])
		print 'mean: ', np.mean(abs_targets[:, 1])


		self.num_batches = len(self.images)/self.batch_size

	def next_batch(self):
		if self.pointer < self.num_batches:
			start = self.pointer * self.batch_size
			end = start + self.batch_size
			self.pointer += 1
			return (self.images[start:end], self.targets[start:end])
		else:
			print 'No more batches'

	def reset_pointer(self):
		self.pointer = 0

# dataloader = DataLoader()
# print dataloader.next_batch()