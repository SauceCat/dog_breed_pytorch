
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os

from imgaug import augmenters as iaa
import numpy as np
import pandas as pd


class DogBreedDataset(Dataset):
	"""Dog Breed dataset."""

	def __init__(self, csv_file, root_dir, transform=None, train=True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		self.train = train

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		im_path = os.path.join(self.root_dir, self.data.iloc[idx]['id'] + '.jpg')
		image = Image.open(im_path)

		if self.train:
			data_sample = {
				'image': image,
				'id': self.data.iloc[idx]['id'],
				'label': self.data.iloc[idx]['breed_label'],
				'label_name': self.data.iloc[idx]['breed']
			}
		else:
			data_sample = {
				'image': image,
				'id': self.data.iloc[idx]['id']
			}

		if self.transform:
			data_sample['image'] = self.transform(image)

		return data_sample


class TrainTransform(object):
	def __init__(self, scale_size, crop_size):
		self.aug = iaa.Sequential([
			iaa.Scale((scale_size, scale_size)),
			iaa.Sometimes(0.25, iaa.OneOf([
				iaa.GaussianBlur(sigma=(0, 3.0)),
				iaa.AverageBlur(k=(1, 5)),
				iaa.MedianBlur(k=(1, 5))
			])),
			iaa.Fliplr(p=0.5),
			iaa.Affine(rotate=(-20, 20), mode='symmetric'),
			iaa.Sometimes(0.25, iaa.OneOf([
				iaa.Dropout(p=(0, 0.1)),
				iaa.Add((-20, 20)),
				iaa.SaltAndPepper(p=0.01),
			])),
			iaa.Sometimes(0.25, iaa.OneOf([
				iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
				iaa.Grayscale(alpha=(0, 1.0))
			])),
			iaa.GammaContrast(gamma=(0.5, 1.5))
		])

		self.crop_size = crop_size

	def __call__(self, img):
		img = np.array(img)
		img_aug = self.aug.augment_image(img)
		img_aug = Image.fromarray(img_aug)

		tfs = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(self.crop_size),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

		img_aug = tfs(img_aug)

		return img_aug


def get_test_time_transform(scale_size, crop_size):
	test_time_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(scale_size),
		torchvision.transforms.CenterCrop(crop_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	return test_time_transform


def get_train_time_transform_simple(scale_size, crop_size):
	train_time_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(scale_size),
		torchvision.transforms.RandomCrop(crop_size),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	return train_time_transform
