
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


def plot_image_grid(image_samples, ncols=4, train=True):
	nrows = np.ceil(len(image_samples) * 1.0 / ncols).astype('int')
	fig_width = 16
	fig_height = int(fig_width * 1.0 / ncols * nrows)

	fig = plt.figure(1, (fig_width, fig_height))
	grids = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0)

	if train:
		im_temp = 'data/train/%s.jpg'
	else:
		im_temp = 'data/test/%s.jpg'
	im_size = (300, 300)

	for item_idx, item in enumerate(image_samples.to_dict('record')):
		img = Image.open(im_temp % item['id']).resize(im_size)
		grids[item_idx].imshow(img)

		grid_text = item['breed'] if train else 'unknown'
		grids[item_idx].text(
			0.5, 0.05, grid_text, verticalalignment='bottom', horizontalalignment='center',
			transform=grids[item_idx].transAxes, color='white', fontsize=12,
			bbox={'facecolor': 'black', 'pad': 5})
		grids[item_idx].set_xticks([])
		grids[item_idx].set_yticks([])

	return grids


def show_data_samples(sample_indices, dataset):
	_ = plt.figure(figsize=(16, 16))
	for i, idx in enumerate(sample_indices):
		data_sample = dataset[idx]

		ax = plt.subplot(4, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(idx))
		ax.imshow(data_sample['image'])
		ax.text(
			0.5, 0.05, data_sample['label_name'],
			verticalalignment='bottom', horizontalalignment='center',
			transform=ax.transAxes, color='white', fontsize=12, bbox={'facecolor': 'black', 'pad': 5})
		ax.axis('off')


def show_data_batch(data_batch):
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])

	_ = plt.figure(figsize=(16, 16))
	for i in np.arange(16):
		ax = plt.subplot(4, 4, i + 1)
		plt.tight_layout()

		im = data_batch['image'][i]
		im = im.numpy().transpose((1, 2, 0))
		im = std * im + mean
		im = np.clip(im, 0, 1)

		ax.imshow(im)
		ax.text(
			0.5, 0.05, data_batch['label_name'][i],
			verticalalignment='bottom', horizontalalignment='center',
			transform=ax.transAxes, color='white', fontsize=12, bbox={'facecolor': 'black', 'pad': 5})
		ax.axis('off')



