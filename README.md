# Dog Breed Identification

by [Li Jiangchun](https://github.com/SauceCat) for NTU Open Source Society

This workshop assumes basic knowledge of Python, Image Recognition and pytorch.

## Setup
Please go to [this folder](https://drive.google.com/drive/folders/1F1uq-ZpY9QlfI-pVE2CiAydEqIIAjFKj?usp=sharing), open `setup.ipynb` with `colab`, and run through the set up process. You can refer to the [setup part](https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop#03-initial-setup) from this post. Thanks [Tu Anqi](https://github.com/anqitu) for providing this.

To test whether you have completed the initial setup properly, try to run `data_explore.ipynb` with `colab`.


## Schedule
1. [data explore](https://github.com/SauceCat/dog_breed_pytorch/blob/master/data_explore.ipynb)
2. [data preparation](https://github.com/SauceCat/dog_breed_pytorch/blob/master/data_preparation.ipynb)
3. [dataset, data loader and data augmentation](https://github.com/SauceCat/dog_breed_pytorch/blob/master/data_loader_augmentation.ipynb)
4. [fine tune a pretrained model](https://github.com/SauceCat/dog_breed_pytorch/blob/master/transfer_learning.ipynb)
5. [bottle-neck features logistic regression](https://github.com/SauceCat/dog_breed_pytorch/blob/master/bottleneck_features_lr.ipynb)
6. visualize training process with TensorBoard
7. other techniques to improve a image recognition model: 
    - test time augmentation
    - ensembling
    - adding more data
    - etc.

## Performance tracking (log loss)
- incep V3, sgd, 0.001, 10 epochs: 0.73262
- incep V3, adam, 0.0001, 10 epochs: 0.70818
- incep V3, adam, 0.0001, 10 epochs, test aug 10 times: 0.68884
- incep V3, adam, 0.001, 10 epochs: 0.51537
- incep V3, adam, 0.0005, 10 epochs: 0.47954
- incep v3, bottle neck, logistic regression: 0.25845
- incep v3, bottle neck, logistic regression, test aug 5 times: 0.22451
- resnet 152, bottle neck, logistic regression: 0.39105
- resnet 152, bottle neck, logistic regression, test aug 5 times: 0.30958
- resnet 152, bottle neck, logistic regression, test aug 10 times: 0.29793
- inception v3 best + restnet 152: 0.22299

**previous best**: xception + inception v3 + incep res: 0.17898
