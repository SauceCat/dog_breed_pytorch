
from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, feature_extract):
	if feature_extract:
		for param in model.parameters():
			param.requires_grad = False


def get_model_input_size(model_name):
	if 'inception' in model_name:
		return 299
	else:
		return 224


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, only_bn=False):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None

	if 'resnet' in model_name:
		# Resnet family

		resnet_model_mapping = {
			'resnet18': models.resnet18(pretrained=use_pretrained),
			'resnet34': models.resnet34(pretrained=use_pretrained),
			'resnet50': models.resnet50(pretrained=use_pretrained),
			'resnet101': models.resnet101(pretrained=use_pretrained),
			'resnet152': models.resnet152(pretrained=use_pretrained)
		}

		model_ft = resnet_model_mapping[model_name]
		set_parameter_requires_grad(model_ft, feature_extract)

		if only_bn:
			model_ft.fc = Identity()
		else:
			# reshape the network
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs, num_classes)

	elif model_name == 'alexnet':
		# Alexnet
		model_ft = models.alexnet(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		# reshape the network
		# (classifier): Sequential(
		# 	...
		# 	(6): Linear(in_features=4096, out_features=1000, bias=True)
		# )
		if only_bn:
			model_ft.classifier[6] = Identity()
		else:
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

	elif 'densenet' in model_name:
		# Densenet family

		densenet_model_mapping = {
			'densenet121': models.densenet121(pretrained=use_pretrained),
			'densenet161': models.densenet161(pretrained=use_pretrained),
			'densenet169': models.densenet169(pretrained=use_pretrained),
			'densenet201': models.densenet201(pretrained=use_pretrained)
		}

		model_ft = densenet_model_mapping[model_name]
		set_parameter_requires_grad(model_ft, feature_extract)
		# reshape the network
		# (classifier): Linear(in_features=1024, out_features=1000, bias=True)
		if only_bn:
			model_ft.classifier = Identity()
		else:
			num_ftrs = model_ft.classifier.in_features
			model_ft.classifier = nn.Linear(num_ftrs, num_classes)

	elif 'vgg' in model_name:
		# vgg family

		vgg_model_mapping = {
			'vgg11': models.vgg11(pretrained=use_pretrained),
			'vgg11_bn': models.vgg11_bn(pretrained=use_pretrained),
			'vgg13': models.vgg13(pretrained=use_pretrained),
			'vgg13_bn': models.vgg13_bn(pretrained=use_pretrained),
			'vgg16': models.vgg16(pretrained=use_pretrained),
			'vgg16_bn': models.vgg16_bn(pretrained=use_pretrained),
			'vgg19': models.vgg19(pretrained=use_pretrained),
			'vgg19_bn': models.vgg19_bn(pretrained=use_pretrained)
		}

		model_ft = vgg_model_mapping[model_name]
		set_parameter_requires_grad(model_ft, feature_extract)
		# reshape the network
		# (classifier): Sequential(
		# 	...
		# 	(6): Linear(in_features=4096, out_features=1000, bias=True)
		# )

		if only_bn:
			model_ft.classifier[6] = Identity()
		else:
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

	elif 'squeezenet' in model_name:
		# Squeezenet family

		squeezenet_model_mapping = {
			'squeezenet1_0': models.squeezenet1_0(pretrained=use_pretrained),
			'squeezenet1_1': models.squeezenet1_1(pretrained=use_pretrained)
		}

		model_ft =squeezenet_model_mapping[model_name]
		set_parameter_requires_grad(model_ft, feature_extract)
		# reshape the network
		# (classifier): Sequential(
		# 	(0): Dropout(p=0.5)
		# 	(1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
		# 	(2): ReLU(inplace)
		# 	(3): AvgPool2d(kernel_size=13, stride=1, padding=0)
		# )
		if only_bn:
			model_ft.classifier[1] = Identity()
		else:
			model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
			model_ft.num_classes = num_classes

	elif 'inception' in model_name:
		# Inception v3
		# Be careful, expects (299,299) sized images and has auxiliary output

		model_ft = models.inception_v3(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		# reshape the model
		# Handle the auxilary net
		if only_bn:
			model_ft.AuxLogits.fc = Identity()
			model_ft.fc = Identity()
		else:
			num_ftrs = model_ft.AuxLogits.fc.in_features
			model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
			# Handle the primary net
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs, num_classes)

	else:
		print("Invalid model name, exiting...")
		exit()

	return model_ft
