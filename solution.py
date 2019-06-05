############
## package loading
############


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import time
import copy
import h5py
import scipy
from scipy import io




import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F



from timeit import default_timer as timer
import pandas as pd
from PIL import Image as Img
import tqdm
from sys import getsizeof

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score

from pathlib import Path
import logging
import datetime as dt
import sys

import itertools

from fastai import *
from fastai.vision import *
import copy



use_gpu = torch.cuda.is_available()
# use_gpu = False
if use_gpu:
    print("Using CUDA")





############
## some functions and classes
############



##### Senext model


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    

    
def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'],         'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    
    
    
pretrained_settings = {

    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}




def se_resnext50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
#     if pretrained is not None:
#         settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
#         initialize_pretrained_model(model, num_classes, settings)
    return model




def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


##### CV folds maker - https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/make_folds.py

def make_folds(_df, n_folds: int) -> pd.DataFrame:
    df = _df.copy()
    cls_counts = Counter(cls for classes in df['attribute_ids_1'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids_1.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids_1.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df



##### Model saving

def save_checkpoint(model, path, if_optimizer=False, if_fastai=False):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """
    if if_fastai==False:
        
        model_name = path.split('-')[0]
        assert (model_name in ['vgg16', 'resnet50', 'lenet','vgg16multi',
                               ]), "Path must have the correct model name"
    
        # Basic details
        checkpoint = {
            #'class_to_idx': model.class_to_idx,
            #'idx_to_class': model.idx_to_class,
            'epochs': model.epochs,
            'history':model.history,
        }

        # Extract the final classifier and the state dictionary
        if model_name == 'vgg16':

            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

        else:

            checkpoint['state_dict'] = model.state_dict()

        if if_optimizer:
            # Add the optimizer
            checkpoint['optimizer'] = model.optimizer
            checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, path)
        
    else:
        
        
        model.save(path, with_opt=True)


##### Model loading

def load_checkpoint(path, model=None, if_checkpoint=True, if_optimizer=False, if_finetune=False, if_fastai=True):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """
    
    # Get the model name
    model_name = path.split('/')[-1].split('-')[0]
    


    if if_checkpoint:
        # Load in checkpoint
        checkpoint = torch.load(path)

    if model is None:

        if model_name =='vgg16':
            model = models.vgg16(pretrained=False)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = checkpoint['classifier']
        elif model_name == 'vgg16multi':
            model = vgg_multi_pretrain(models.vgg16(pretrained=False))
            if if_finetune == False:
                for param in model.features.parameters():
                    param.requires_grad = False
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            if if_finetune == False:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=2048,
                            out_features=1101
                        ),
                        #torch.nn.Sigmoid()
                    )
        elif model_name.startswith('se_resnext50'):
            model = se_resnext50()
            weights = torch.load(path)
            model.load_state_dict(weights)
            if if_finetune == False:
                for param in model.parameters():
                    param.requires_grad = False
            
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
            
            model.last_linear = torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=2048,
                            out_features=1101
                        ),
                        #torch.nn.Sigmoid()
                    )
            print('pretrained se_resnext50 weights have been loaded')



    if if_checkpoint:
        if if_fastai:
            model.load_state_dict(checkpoint['model'])
        else:# Load in the state dict
            model.load_state_dict(checkpoint['state_dict'])
            # Model basics
            model.epochs = checkpoint['epochs']
            model.history = checkpoint['history']
    else:
        model.epochs = 0
        model.history=[]


    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if use_gpu:
        model = model.to('cuda')



    if if_optimizer:
        # Optimizer
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    else:

        return model



############
## settings
############


BATCH_SIZE = 64
SIZE = 288


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = get_logger(name="Main", tag="Pytorch-Resnet50")



TRAIN_DIR = '../input/imet-2019-fgvc6/train/'
TEST_DIR = '../input/imet-2019-fgvc6/test/'
MODEL_DIR = '../input/pretrained_zoo/'
MODEL_NAME = 'se_resnext50_32x4d-a260b3a4.pth'
INPUT_DIR = '../input/imet-2019-fgvc6/'


df_train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
df_train['attribute_ids_1'] = df_train['attribute_ids'].copy()
df_train['attribute_ids'] = df_train['attribute_ids'].str.split(" ")


df_test = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))


train_name=df_train['id'].unique().tolist()


############
## trainer and callback
############

class Trainer:
    def __init__(self,  
                 criterion,
                 train_dir,
                 logger,
                 model=None,
                 ):
        self.model = model
        self.criterion = criterion
        self.train_dir = train_dir
        self.logger = logger

        
        
    def fit(self, lr, train_name, save_file_name, 
            n_epochs=5, 
            warm_start=None,
            tot_epochs=None,
            start_epoch=None,
            one_cycle=True,
            find_lr=False):
        train_preds = np.zeros((len(train_name), 1103))

        model = self.model
    
        # make 5 folds
        tmp_df_train = make_folds(df_train.iloc[0:len(train_name)],5)
        print(tmp_df_train.groupby(['fold']).size())
        
        # specific which fold to train(fold 1 in this case)
        for i in range(1):
            trn_idx = np.where( tmp_df_train['fold']!=i)[0]
            val_idx = np.where( tmp_df_train['fold']==i)[0]
    

            self.fold_num = i
            
            
            if model is None:
                # create dummy cnn_learner to get processed model from fastai
                tmp_tfms = get_transforms()

                tmp_train = ImageList.from_df(df_train, 
                              path=INPUT_DIR, cols='id', folder='train', suffix='.png')

                tmp_data_bunch = (tmp_train.split_by_idxs(train_idx=trn_idx, valid_idx=val_idx)
                    .label_from_df(cols='attribute_ids')
                    .transform(tmp_tfms, size=SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
                    .databunch(path=Path('.'), bs=BATCH_SIZE).normalize(imagenet_stats))

                #tmp_lr = cnn_learner(data=tmp_data_bunch, base_arch=models.resnet50, loss_func=self.criterion, metrics=fbeta)
                tmp_lr = cnn_learner(data=tmp_data_bunch, base_arch=se_resnext50, loss_func=self.criterion, metrics=fbeta)

                model = copy.deepcopy(tmp_lr.model)
                tmp_lr.destroy()
            

            # Create data_bunch for training
            tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, 
                          max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
             xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),])
    
            
            train, test = [ImageList.from_df(df, path=INPUT_DIR, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([df_train, df_test], ['train', 'test'])]
            
            data_bunch = (train.split_by_idxs(train_idx=trn_idx, valid_idx=val_idx)
                .label_from_df(cols='attribute_ids')
                .add_test(test)
                .transform(tfms, size=SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
                .databunch(path=Path('.'), bs=BATCH_SIZE).normalize(imagenet_stats))
            
            
            learn = Learner(data_bunch, model, loss_func=self.criterion, metrics=fbeta, model_dir='.')
        
            if type(lr)==slice:
                learn.split(lambda m: (m[1]))
                
            learn.unfreeze()
            
            
            
            if warm_start is not None:
                learn.load(warm_start)
            
            
            cb = archiver(learn, 
                           criterion=self.criterion, 
                           save_file_name=save_file_name,
                           logger = self.logger)

            if find_lr:
                learn.lr_find()
                learn.recorder.plot(suggestion=True)
                learn.recorder.plot_lr()
                learn.recorder.plot_losses()
            
            else:
                if one_cycle:
                    learn.fit_one_cycle(n_epochs, lr, callbacks=[cb], tot_epochs=tot_epochs, start_epoch=start_epoch)

                else:
                    learn.fit(n_epochs, lr, callbacks=[cb])
            
        
        
        
        return learn
    
    

    

class archiver(LearnerCallback):
    def __init__(self, learn:Learner, criterion, save_file_name, logger):
        super().__init__(learn)
        self.criterion = criterion
        self.save_file_name = save_file_name
        self.valid_loss_min = np.Inf
        self.logger = logger
        self.best_score = 0.0

        
    "Update a graph of learner stats and metrics after each epoch."
    def on_epoch_end(self, epoch:int, smooth_loss:Tensor,last_metrics, **kwargs):


        # Logging
        logger.info("=========================================")
        logger.info(f"Epoch {epoch}")
        logger.info("=========================================")
        logger.info(f"train_loss: {smooth_loss.item():.8f};")
        logger.info(last_metrics)
        #logger.info(f"val_f2score: {valid_f2:.8f}")
        
        
        
        # Track min validation loss
        if last_metrics[1].item() > self.best_score:
            self.best_score = last_metrics[1].item()
            logger.info(f"best score is now: {last_metrics[1].item():.8f};")

            # Validation predictions
            valid_preds = self.learn.get_preds(DatasetType.Valid)
            np.save('valid_preds', valid_preds[0])
            np.save('valid_tgts', valid_preds[1])
            
        
            
            # Save model
            save_checkpoint(self.learn, path=self.save_file_name, if_fastai=True)

            # Save learner
            self.learn.export(self.save_file_name)
        
        
            

############
## main
############

model = load_checkpoint(os.path.join(MODEL_DIR, MODEL_NAME), if_checkpoint=False, if_finetune=True)
criterion = nn.BCEWithLogitsLoss(reduction="mean")


trainer = Trainer(criterion, TRAIN_DIR, logger, model=model)


lr = trainer.fit(train_name=train_name, 
                save_file_name='seresnext50-transfer-1',
                lr=3e-4,
                n_epochs=17,
                one_cycle=True,
                find_lr=False)