
import time
import copy
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import log_loss
from tensorboardX import SummaryWriter

from .utils import recover_image


def train_model(
		model, device, dataloaders, criterion, optimizer, scheduler=None,
		num_epochs=25, is_inception=False, in_notebook=True):
	since = time.time()
	# initialize a progress bar
	tqdm_func = tqdm_notebook if in_notebook else tqdm
	writer = SummaryWriter()

	# initialize the best model
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_log_loss = math.inf

	# initialize a list for storing the performance history
	train_performance_history = []
	val_performance_history = []

	for epoch in tqdm_func(range(num_epochs)):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				# set model to training mode
				if scheduler is not None:
					scheduler.step()
				model.train()
				num_batch_per_epoch = int(
					np.ceil(len(dataloaders['train'].dataset) * 1.0 / dataloaders['train'].batch_size))
			else:
				# set model to evaluation mode
				model.eval()
				num_batch_per_epoch = int(
					np.ceil(len(dataloaders['val'].dataset) * 1.0 / dataloaders['val'].batch_size))

			running_loss = 0.0
			running_corrects = 0

			if is_inception:
				running_loss1 = 0.0
				running_loss2 = 0.0

			# Iterate over data.
			for batch_idx, batch_data in tqdm_func(
					enumerate(dataloaders[phase]), total=num_batch_per_epoch):

				writer_step = num_batch_per_epoch * epoch + (batch_idx + 1)

				inputs = batch_data['image'].to(device)
				labels = batch_data['label'].to(device)
				img = recover_image(inputs[0].cpu(), writer=True)

				writer.add_image('image/transform', img, writer_step)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Get model outputs and calculate loss
					# Special case for inception because in training it has an auxiliary output. In train
					#   mode we calculate the loss by summing the final output and the auxiliary output
					#   but in testing we only consider the final output.
					if is_inception and phase == 'train':
						# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4 * loss2

						writer.add_scalar('loss/loss1', loss1.item(), writer_step)
						writer.add_scalar('loss/loss2', loss2.item(), writer_step)
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					_, preds = torch.max(outputs, 1)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()
						writer.add_scalar('loss/loss', loss.item(), writer_step)

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

				# also track loss1 and loss2 when is inception
				if is_inception:
					running_loss1 += loss1.item() * inputs.size(0)
					running_loss2 += loss2.item() * inputs.size(0)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()
			print('Epoch {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

			# record the history
			train_performance_history.append({
				'epoch': epoch,
				'train_loss': epoch_loss,
				'train_acc': epoch_acc
			})

			if is_inception:
				train_performance_history['train_loss1'] = running_loss1 / len(dataloaders[phase].dataset)
				train_performance_history['train_loss2'] = running_loss2 / len(dataloaders[phase].dataset)

			# deep copy the model
			if phase == 'val':
				val_performance_history.append({
					'epoch': epoch,
					'val_loss': epoch_loss,
					'val_acc': epoch_acc
				})

				writer.add_scalar('loss/val_loss', epoch_loss, writer_step)
				writer.add_scalar('accuracy/val_acc', epoch_acc, writer_step)
				if epoch_loss < best_log_loss:
					best_log_loss = epoch_loss
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
			else:
				writer.add_scalar('accuracy/train_acc', epoch_acc, writer_step)

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best validation Loss: {:4f} Acc: {:.4f}'.format(best_log_loss, best_acc))

	# process the full history
	full_performance_history = pd.DataFrame(train_performance_history).merge(
		pd.DataFrame(val_performance_history), on='epoch', how='left')

	# load best model weights
	model.load_state_dict(best_model_wts)

	writer.close()
	return model, full_performance_history


def apply_model(
		model, device, criterion, data_loader,
		in_notebook=True, val=False, test_time_aug=False, aug_times=5):
	since = time.time()
	# initialize a progress bar
	tqdm_func = tqdm_notebook if in_notebook else tqdm

	# set model to evaluation mode
	model.eval()
	num_batch_per_epoch = int(
		np.ceil(len(data_loader.dataset) * 1.0 / data_loader.batch_size))

	# initialize a dict for storing the prediction results
	prediction_results = dict(
		probs=[],
		labels=[],
		image_ids=[]
	)

	# config test time augmentation
	num_epochs = 1
	if test_time_aug:
		num_epochs = aug_times

	for epoch in tqdm_func(range(num_epochs)):
		# for validation debug
		running_loss = 0.0
		running_corrects = 0

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		torch.set_grad_enabled(False)

		# Iterate over data.
		for batch_idx, batch_data in tqdm_func(
				enumerate(data_loader), total=num_batch_per_epoch):

			inputs = batch_data['image'].to(device)
			ids = batch_data['id']

			torch.set_grad_enabled(False)
			outputs = model(inputs)
			probs = torch.nn.Softmax()(outputs)
			_, preds = torch.max(outputs, 1)

			if val:
				labels = batch_data['label'].to(device)
				loss = criterion(outputs, labels)

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
				prediction_results['labels'] += list(labels.cpu().numpy())

			prediction_results['probs'] += list(probs.cpu().numpy())
			prediction_results['image_ids'] += ids

		if val:
			epoch_loss = running_loss / len(data_loader.dataset)
			epoch_acc = (running_corrects.double() / len(data_loader.dataset)).item()
			print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	print()

	scores_all = np.array(prediction_results['probs'])
	scores_avg = scores_all[:len(data_loader.dataset)]

	if num_epochs > 1:
		for i in range(1, num_epochs):
			scores_avg += scores_all[i*len(data_loader.dataset):(i+1)*len(data_loader.dataset)]

		scores_avg = scores_avg / num_epochs

	if val:
		print('debug log loss using sklearn...')
		for i in range(num_epochs):
			epoch_loss = log_loss(
				y_true=prediction_results['labels'][i*len(data_loader.dataset):(i+1)*len(data_loader.dataset)],
				y_pred=scores_all[i*len(data_loader.dataset):(i+1)*len(data_loader.dataset)])
			print('epoch {} loss: {:.4f}'.format(i, epoch_loss))

		val_labels = np.array(prediction_results['labels'][:len(data_loader.dataset)])
		final_loss = log_loss(y_true=val_labels, y_pred=scores_avg)
		print('final loss: {:.4f}'.format(final_loss))

	time_elapsed = time.time() - since
	print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	# process the final results
	final_results = dict(
		image_ids=prediction_results['image_ids'][:len(data_loader.dataset)],
		probs=[list(r) for r in scores_avg]
	)

	return pd.DataFrame(final_results)


def get_model_bottle_neck(
		model, device, data_loader, in_notebook=True, train_or_val=False, test_time_aug=False, aug_times=5):
	since = time.time()

	# initialize a progress bar
	tqdm_func = tqdm_notebook if in_notebook else tqdm

	# set model to evaluation mode
	model.eval()
	num_batch_per_epoch = int(
		np.ceil(len(data_loader.dataset) * 1.0 / data_loader.batch_size))

	# initialize a dict for storing the prediction results
	prediction_results = dict(
		outputs=[],
		image_ids=[]
	)
	if train_or_val:
		prediction_results['labels'] = []

	# config test time augmentation
	num_epochs = 1
	if test_time_aug:
		num_epochs = aug_times

	for epoch in tqdm_func(range(num_epochs)):

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		torch.set_grad_enabled(False)

		# Iterate over data.
		for batch_idx, batch_data in tqdm_func(
				enumerate(data_loader), total=num_batch_per_epoch):

			inputs = batch_data['image'].to(device)
			ids = batch_data['id']
			outputs = model(inputs)

			if train_or_val:
				prediction_results['labels'] += list(batch_data['label'].cpu().numpy())

			prediction_results['outputs'] += list(outputs.cpu().numpy())
			prediction_results['image_ids'] += ids

	print()

	time_elapsed = time.time() - since
	print('Extracting feature complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	return pd.DataFrame(prediction_results)
