
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


def grid_black_bg_text(grid, text):
	_ = grid.text(
		0.5, 0.05, text, verticalalignment='bottom', horizontalalignment='center',
		transform=grid.transAxes, color='white', fontsize=12,
		bbox={'facecolor': 'black', 'pad': 5})


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

		ax = grids[item_idx]
		ax.imshow(img)

		grid_text = item['breed'] if train else 'unknown'
		grid_black_bg_text(ax, grid_text)
		ax.axis('off')

	return grids


def show_data_samples(sample_indices, dataset):
	fig = plt.figure(figsize=(16, 16))
	grids = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)

	for i, idx in enumerate(sample_indices):
		data_sample = dataset[idx]
		ax = grids[i]

		ax.set_title('Sample #{}'.format(idx))
		ax.imshow(data_sample['image'])
		grid_black_bg_text(ax, data_sample['label_name'])
		ax.axis('off')


def show_data_batch(data_batch, figsize=(16, 16), train=True):
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])

	fig = plt.figure(figsize=figsize)
	grids = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)

	for i in np.arange(16):

		im = data_batch['image'][i]
		im = im.numpy().transpose((1, 2, 0))
		im = std * im + mean
		im = np.clip(im, 0, 1)

		ax = grids[i]
		ax.imshow(im)
		grid_text = data_batch['label_name'][i] if train else 'unknown'
		grid_black_bg_text(ax, grid_text)
		ax.axis('off')
