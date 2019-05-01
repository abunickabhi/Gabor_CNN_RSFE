#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:03:09 2019

@author: abhi
"""
import os
import keras
import spectral
import numpy as np
from scipy.io import loadmat 
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils


def sample_gt(gt, percentage):
    mask = np.zeros(gt.shape, dtype='bool')
    for l in np.unique(gt):
       x, y = np.nonzero(gt == l)
       indices = np.random.choice(len(x), int(len(x) * percentage),replace=False)
       x, y = x[indices], y[indices]
       mask[x, y] = True
       train_gt = np.zeros_like(gt)
       train_gt[mask] = gt[mask]
       test_gt = np.zeros_like(gt)
       test_gt[~mask] = gt[~mask]
    return train_gt, test_gt

def get_model(name, **kwargs):
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = np.ones(n_classes)
    weights = kwargs.setdefault('weights', weights)
    #batch_size = [10, 20, 40, 60, 80, 100] for doing grid search over the CNN network, batch size is one of the most important hyperparameter
    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model,model_name = Baseline(n_bands, n_classes,kwargs.setdefault('dropout', True))
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optimizers.Adam(lr=lr)
        criterion = 'categorical_crossentropy'
        kwargs.setdefault('epoch', 50)
        kwargs.setdefault('batch_size', 300)

    epoch = kwargs.setdefault('epoch', 50)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('n_classes', 20)
    kwargs['center_pixel'] = center_pixel
    return model,model_name, optimizer, criterion, kwargs


def Baseline(input_shape, num_classes, dropout=True):
    model_name='Baseline'
    model = Sequential()
    model.add(Dense(2048, input_dim=input_shape, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))    
    model.add(Dense(2048, activation='relu'))    
    model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation='softmax'))
    return model,model_name

def load_data(data, gt,**hyperparams):
    patch=[]
    patch_label=[]
    n_classes=hyperparams['n_classes']
    patch_size = hyperparams['patch_size']
    ignored_labels = set(hyperparams['ignored_labels'])
    center_pixel = hyperparams['center_pixel']
    supervision = hyperparams['supervision']
    if supervision == 'full':
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
    x_pos, y_pos = np.nonzero(mask)
    p = patch_size // 2
    indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
     
    for i in range(len(indices)):
        x, y = indices[i]
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size

        xx = data[x1:x2, y1:y2]
        yy = gt[x1:x2, y1:y2]
        xx = np.asarray(np.copy(xx).transpose((2, 0, 1)), dtype='float32')
        yy = np.asarray(np.copy(yy), dtype='int64')
        # Extract the center label if needed
        if center_pixel and patch_size > 1:
            yy = yy[patch_size // 2, patch_size // 2]
        
        # Remove unused dimensions when we work with invidual spectrums
        elif patch_size == 1:
            xx = xx[:, 0, 0]
            yy = yy[0, 0]
        if patch_size > 1:
            xx = xx.reshape(1,xx.shape[0], xx.shape[1], xx.shape[2])           
        patch.append(xx)        
        patch_label.append(yy)
        
    X_test=np.asarray(patch) 
    y_test=np.asarray(patch_label)
    if supervision == 'semi':
        data_loss=np.asarray(patch)[:,:,:,patch_size//2,patch_size//2].squeeze() 
        y_true=keras.utils.to_categorical(y_test-1, n_classes)
        
        target=[y_true,data_loss]
    else:
        if not center_pixel and patch_size > 1:
            target=keras.utils.to_categorical(y_test-1, n_classes).transpose(0,3,1,2)
            
        else:
            target=keras.utils.to_categorical(y_test-1, n_classes)    
    return X_test, target
    

def flip(*arrays):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        arrays = [np.fliplr(arr) for arr in arrays]
    if vertical:
        arrays = [np.flipud(arr) for arr in arrays]
    return arrays


def batch_iter(data, gt, shuffle, **hyperparams):
    name = hyperparams['dataset']
    n_classes=hyperparams['n_classes']
    patch_size = hyperparams['patch_size']
    batch_size=hyperparams['batch_size']
    ignored_labels = set(hyperparams['ignored_labels'])
    center_pixel = hyperparams['center_pixel']
    supervision = hyperparams['supervision']
    if supervision == 'full':
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
    x_pos, y_pos = np.nonzero(mask)
    p = patch_size // 2
    indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
    labels = [gt[x,y] for x,y in indices]
    num_batches_per_epoch = int((len(indices) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(indices)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                np.random.shuffle(indices)            
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                patch=[]
                patch_label=[]
                indexes=indices[start_index:end_index] 
                for i in range(len(indexes)):
                    x, y = indexes[i]
                    x1, y1 = x - patch_size // 2, y - patch_size // 2
                    x2, y2 = x1 + patch_size, y1 + patch_size            
                    xx = data[x1:x2, y1:y2]
                    yy = gt[x1:x2, y1:y2]            
                    # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
                    xx = np.asarray(np.copy(xx).transpose((2, 0, 1)), dtype='float32')
                    yy = np.asarray(np.copy(yy), dtype='int64')            
                    # Extract the center label if needed
                    if center_pixel and patch_size > 1:
                        yy = yy[patch_size // 2, patch_size // 2]
                    # Remove unused dimensions when we work with invidual spectrums
                    elif patch_size == 1:
                        xx = xx[:, 0, 0]
                        yy = yy[0, 0]
                    if patch_size > 1:
                        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                        xx = xx.reshape(1,xx.shape[0], xx.shape[1], xx.shape[2])
                    patch.append(xx)
                    patch_label.append(yy)                    
                X=np.asarray(patch) 
                y=np.asarray(patch_label)
                
               
                if not center_pixel and patch_size > 1:
                    target=keras.utils.to_categorical(y-1, n_classes).transpose(0,3,1,2)
                else:
                    target=keras.utils.to_categorical(y-1, n_classes)
            yield X, target

    return num_batches_per_epoch, data_generator()



class HyperX(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, gt, **hyperparams):
        'Initialization'
        self.data = data
        self.label = gt
        self.n_classes=hyperparams['n_classes']
        self.batch_size=hyperparams['batch_size']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        self.supervision = hyperparams['supervision']
        if self.supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0     
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]         
        self.on_epoch_end()

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.indices) - 1) / self.batch_size) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        patch=[]
        patch_label=[]
        for i in range(len(indexes)):
            x, y = indexes[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
    
            data = self.data[x1:x2, y1:y2]
            label = self.label[x1:x2, y1:y2]
    
            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.1:
                    data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.2:
                    data = self.mixture_noise(data, label)
    
            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')    
            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data = data.reshape(1,data.shape[0], data.shape[1], data.shape[2])
            patch.append(data)
            patch_label.append(label)        
        X=np.asarray(patch) 
        y=np.asarray(patch_label)         
        target=keras.utils.to_categorical(y-1, self.n_classes)            
        return X, target
    
    
    
def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir)
         filename = str(datetime.datetime.now())
         joblib.dump(model, model_dir + filename + '.pkl')

def test(model, img, hyperparams):
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size = hyperparams['batch_size']
    n_classes = hyperparams['n_classes']
    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in grouper(batch_size, sliding_window(img, **kwargs)):
        if patch_size == 1:
            data = [b[0][0, 0] for b in batch]
            data = np.copy(data)
        else:
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data=np.expand_dims(data,axis=1)
        indices = [b[1:] for b in batch]
        output=model.predict(data)
        if isinstance(output, list):
            output = output[0]
        for (x, y, w, h), out in zip(indices, output):
            if center_pixel:
                probs[x + w // 2, y + h // 2] += out
            else:
                probs[x:x + w, y:y + h] += out
    return probs
   
