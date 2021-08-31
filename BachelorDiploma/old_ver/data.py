import os
import numpy as np
import glob
import cv2

import keras
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans

Sky = [0, 0, 0]
Other = [255, 255, 255]
# Building = [128, 0, 0]
# Pole = [192, 192, 128]
# Road = [128, 64, 128]
# Pavement = [60, 40, 222]
# Tree = [128, 128, 0]
# SignSymbol = [192, 128, 128]
# Fence = [64, 64, 128]
# Car = [64, 0, 128]
# Pedestrian = [64, 64, 0]
# Bicyclist = [0, 128, 192]
# Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([
		Sky,
		Other
	])


def mirror_boards(image, axis=0, board_size=32):
	if axis == 0:
		f_board = image[:board_size, :, :]
		s_board = image[image.shape[axis] - board_size:, :, :]
	else:
		f_board = image[:, :board_size, :]
		s_board = image[:, image.shape[axis] - board_size:, :]

	f_board = cv2.flip(f_board, axis)
	s_board = cv2.flip(s_board, axis)

	if len(f_board.shape) == 2:
		f_board = np.expand_dims(f_board, 2)
		s_board = np.expand_dims(s_board, 2)

	# print('f_board : {0} ; s_board : {1} image : {2}'.format(f_board.shape, s_board.shape, image.shape))
	return np.concatenate([f_board, image, s_board], axis)


def adjust_image(image, board_size=32):
	#     print(image.shape)
	image = mirror_boards(image, board_size=board_size)
	image = mirror_boards(image, axis=1, board_size=board_size)

	image = cv2.resize(image, (256, 256))

	if len(image.shape) == 2:
		image = np.expand_dims(image, 2)

	return image


def adjust_images(images):
	images = [adjust_image(image) for image in images]
	images = np.array(images)

	return images

def adjustData(
		img,
		mask,
		flag_multi_class,
		num_class,
		mirror):
	if(flag_multi_class):
		img = img / 255
		mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
		new_mask = np.zeros(mask.shape + (num_class,))
		for i in range(num_class):
			new_mask[mask == i, i] = 1
		new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
		mask = new_mask

	elif(np.max(img) > 1):
		flag = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
		if flag > 0.5 and mirror:
			img = adjust_images(img)
			mask = adjust_images(mask)

		img = img / 255
		mask = mask / 255
# 		rev_mask = mask.copy()
		mask[mask > 0.5] = 1
		mask[mask <= 0.5] = 0
# 		mask = np.concatenate([cv2.bitwise_not(mask), mask], axis=-1)

	return (img, mask)


class BatchGenerator(keras.utils.Sequence):

	def __init__(
			self,
			batch_size,
			train_path,
			image_folder,
			mask_folder,
			aug_dict,
			image_color_mode="grayscale",
			mask_color_mode="grayscale",
			image_save_prefix="image",
			mask_save_prefix="mask",
			flag_multi_class=False,
			num_class=2,
			save_to_dir=None,
			target_size=(256, 256),
			seed=1,
			shuffle=True,
			test_gen = False):

		image_datagen = ImageDataGenerator(**aug_dict)
		mask_datagen = ImageDataGenerator(**aug_dict)
		image_generator = image_datagen.flow_from_directory(
			train_path,
			classes=[image_folder],
			class_mode=None,
			color_mode=image_color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=image_save_prefix,
			seed=seed,
			shuffle=shuffle)
		mask_generator = mask_datagen.flow_from_directory(
			train_path,
			classes=[mask_folder],
			class_mode=None,
			color_mode=mask_color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=mask_save_prefix,
			seed=seed,
			shuffle=shuffle)

		self.image_generator = image_generator
		self.mask_generator = mask_generator
		self.is_test = test_gen
		
		n_images = len(os.listdir(os.path.join(train_path, image_folder)))
		self.n_batches = n_images // batch_size

		self.flag_multi_class = flag_multi_class
		self.num_class = num_class
        
		self.mirror = True
		if image_color_mode == 'grayscale' or test_gen:
			self.mirror = False

	def __len__(self):

		return self.n_batches

	def __getitem__(self, index):
		img = self.image_generator[index]
		mask = img
		if not self.is_test:
			mask = self.mask_generator[index]#[:,:,:,0]
# 		print(np.shape(mask))
        
		img, mask = adjustData(img, mask, self.flag_multi_class, self.num_class, mirror=self.mirror)

		return (img, mask)


def testGenerator(test_path,target_size = (256,256),flag_multi_class = False,as_gray = True):
	test_images = glob.glob(os.path.join(test_path, '*'))
	test_images.sort()
	for path in test_images:
		img = cv2.imread(path,0)
		img = img / 255
		img = cv2.resize(img,target_size)
		img = np.expand_dims(img, 2)
		img = np.expand_dims(img, 0)
		yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
	image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
	image_arr = []
	mask_arr = []
	for index,item in enumerate(image_name_arr):
		img = io.imread(item,as_gray = image_as_gray)
		img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
		mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
		mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
		img,mask = adjustData(img,mask,flag_multi_class,num_class)
		image_arr.append(img)
		mask_arr.append(mask)
	image_arr = np.array(image_arr)
	mask_arr = np.array(mask_arr)
	return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
	img = img[:,:,0] if len(img.shape) == 3 else img
	img_out = np.zeros(img.shape + (3,))
	for i in range(num_class):
		img_out[img == i,:] = color_dict[i]
	return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2, suf=''):
	for i,item in enumerate(npyfile):
		img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
		io.imsave(os.path.join(save_path,"{0}{1}.png".format(i, suf)),img)
