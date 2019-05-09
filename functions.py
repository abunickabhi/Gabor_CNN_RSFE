"""
Created on Wed May  1 10:03:09 2019

@author: abhi

This file contains all the necessary functions for implementation of CNN
models and gabor filtering
"""
import os
import keras
import spectral
import numpy as np
from scipy.io import loadmat 
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Lambda
from keras.layers.core import Lambda
from keras.layers import Reshape
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import BatchNormalization
from keras import optimizers, regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Convolution3D, MaxPooling3D,Conv3D,ZeroPadding3D
from keras.layers import Activation, Dense,Dropout,Flatten

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
    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 20)
        kwargs.setdefault('batch_size', 100)
        center_pixel = True
        input_channels=((kwargs['batch_size'],n_bands))
        model, model_name = HuEtAl.build(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optimizers.Adam(lr=lr)
        criterion = 'categorical_crossentropy'
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        #n_bands=12
        #n_classes=20
        model,model_name = HamidaEtAl.build(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0005)
        epoch = kwargs.setdefault('epoch', 50)
        criterion = 'categorical_crossentropy'

    epoch = kwargs.setdefault('epoch', 50)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('n_classes', 20)
    kwargs['center_pixel'] = center_pixel
    return model,model_name, optimizer, criterion, kwargs


def Baseline(input_shape, num_classes, dropout=True):
    model_name='Baseline'
    model = Sequential()
    model.add(Dense(20, input_dim=input_shape, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))    
    model.add(Dense(20, activation='relu'))    
    model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation='softmax'))
    return model,model_name


def _2DCNN(input_shape, num_classes):
    """
    This function will operate on a 19 class dataset which is very sparse so the basic idea is to divide it into training 
    and testing mask.
    The model tensor has to be flattened from the training starts.    
    """
    model_name = 'nn'
    model = Sequential()
    model.add(Conv2D(64, (5, 5),input_shape=input_shape, activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model,model_name

class HuEtAl:
    @staticmethod
    def build(input_channels, n_classes, kernel_size=None, pool_size=None):
        if kernel_size is None:
           kernel_size = input_channels // 10 + 1
        if pool_size is None:
           pool_size = kernel_size // 5 + 1
        input_x=Input(shape=(input_channels,))
        x = Lambda(expand_dims, expand_dims_output_shape)(input_x)    
        x=Conv1D (20,kernel_size=kernel_size)(x)
        x=MaxPooling1D(pool_size=pool_size)(x)
        x=Activation("relu")(x)
        x=Flatten()(x)
        x=Dense(100, activation='relu')(x)
        x=Dense(n_classes, activation='softmax')(x)
        model_name ='hu'    
        model=Model(input_x,x)
        return model, model_name
    
class LiEtAl():
    @staticmethod
    def build(input_channels, n_classes, n_planes=2, patch_size=5):
        model_name="LiEtAl"
        input_x=Input(shape=(1,input_channels,patch_size,patch_size))
        x = ZeroPadding3D(padding=(1,0,0))(input_x)
        x=Conv3D(n_planes, kernel_size=(7, 3, 3),padding="same")(x)
        x=Activation("relu")(x)
        
        x = ZeroPadding3D(padding=(1,0,0))(x)
        x=Conv3D(2*n_planes, kernel_size=(3, 3, 3),padding="same")(x)
        x=Activation("relu")(x)
        
        x = Flatten()(x)
        x = Dense(n_classes, activation='softmax')(x)
        model = Model(input_x, x)
        return model, model_name

class HamidaEtAl:
    @staticmethod
    def build(input_channels, n_classes, patch_size=5, dilation=1):
                
        patch_size = patch_size
        input_shape = (1,input_channels,patch_size,patch_size)
        dilation = (dilation, 1, 1)        
        model_name="HamidaEtAl"
        model = Sequential()
        if patch_size == 3:
            model.add(Conv3D(20, (3, 3, 3), strides=(1, 1, 1), dilation_rate=dilation, padding="same", input_shape=input_shape))
        else:
            model.add(Conv3D(20, (3, 3, 3), strides=(1, 1, 1), dilation_rate=dilation, padding="same",input_shape=input_shape))
        model.add(Activation("relu"))
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv3D(20, (3, 1, 1), dilation_rate=dilation, strides=(2, 1, 1)))          
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv3D(35, (3, 3, 3), dilation_rate=dilation, strides=(1, 1, 1))) 
        model.add(Activation("relu"))            
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv3D(35, (3, 1, 1), dilation_rate=dilation,activation="relu", strides=(2, 1, 1))) 
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv3D(35, (3, 1, 1), dilation_rate=dilation,activation="relu", strides=(1, 1, 1))) 
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv3D(35, (2, 1, 1), dilation_rate=dilation, strides=(2, 1, 1)))        
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax'))        
        return model, model_name

class Hamida2:
    @staticmethod
    def build(input_channels, n_classes, patch_size=5, dilation=1):
                
        patch_size = patch_size
        input_shape = (1,input_channels,patch_size,patch_size)
        dilation = (dilation, 1, 1)        
        model_name="HamidaEtAl"
        model = Sequential()     
        model.add(Conv2D(20, (3, 3), strides=(1, 1), dilation_rate=dilation, padding="same",input_shape=input_shape))
        model.add(Activation("relu"))        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv2D(20, (3, 1), dilation_rate=dilation, strides=(2, 1)))          
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv2D(35, (3, 3), dilation_rate=dilation, strides=(1, 1))) 
        model.add(Activation("relu"))            
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv2D(35, (3, 1), dilation_rate=dilation,activation="relu", strides=(2, 1))) 
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv2D(35, (3, 1), dilation_rate=dilation,activation="relu", strides=(1, 1))) 
        
        model.add(ZeroPadding3D(padding=(1,0,0)))
        model.add(Conv2D(35, (2, 1), dilation_rate=dilation, strides=(2, 1)))        
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax'))        
        return model, model_name
    
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



class dtimport(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, gt, **hyperparams):
        'Initialization'
        self.data = data
        self.label = gt
        self.n_classes=hyperparams['n_classes']
        self.batch_size=hyperparams['batch_size']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])       
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
   
def gt_change(gt):
    gtr=np.copy(gt)
    row=len(gt[:,1])
    col=len(gt[1,:])
    for i in range(row):
        for j in range(col):
            if gt[i][j]==1:
                gtr[i][j]=1
            elif gt[i][j]==6:
                gtr[i][j]=2
            elif gt[i][j]==7:
                gtr[i][j]=3
            elif gt[i][j]==8:
                gtr[i][j]=4
            elif gt[i][j]==10:
                gtr[i][j]=5
            elif gt[i][j]==12:
                gtr[i][j]=6
            elif gt[i][j]==13:
                gtr[i][j]=7
            elif gt[i][j]==14:
                gtr[i][j]=8
            elif gt[i][j]==15:
                gtr[i][j]=9
            elif gt[i][j]==16:
                gtr[i][j]=10
            elif gt[i][j]==17:
                gtr[i][j]=11
            elif gt[i][j]==22:
                gtr[i][j]=12
            elif gt[i][j]==23:
                gtr[i][j]=13
            elif gt[i][j]==25:
                gtr[i][j]=14
            elif gt[i][j]==26:
                gtr[i][j]=15
            elif gt[i][j]==28:
                gtr[i][j]=16
            elif gt[i][j]==29:
                gtr[i][j]=17
            elif gt[i][j]==30:
                gtr[i][j]=18
            elif gt[i][j]==32:
                gtr[i][j]=19
            elif gt[i][j]==33:
                gtr[i][j]=20
            else:
                gtr[i][j]=0
                
        return gtr
                
def count_sliding_window(top, step=10, window_size=(20, 20)):
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h
                
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
        
def metrics(prediction, target, ignored_labels=[], n_classes=None):
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))
    results["Confusion matrix"] = cm
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa
    return results

def show_results(results, label_values=None, agregated=False):
    text = ""
    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)
    print(text)
    
    
# These functions are taken from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats
def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i
def power(image, kernel):
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)