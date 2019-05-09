import itertools
import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  
from sklearn.metrics import confusion_matrix


class TestCallback(Callback):
    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return


# callback to save training accuracy and loss after every epoch:
class SaveHistory(Callback):

    def __init__(self, save_file):
        self.save_file = save_file
        self.losses = []
        self.acc = []

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        # collect and save history after every epoch:
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        loss_hist = np.array(self.losses)
        acc_hist = np.array(self.acc)
        np_hist = np.vstack((loss_hist, acc_hist))
        np.savetxt(self.save_file, np_hist, delimiter=" ")

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

class AccLossPlotter(Callback):
    def __init__(self, graphs=['acc', 'loss'], save_graph=False):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph


    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.figure()
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        
        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)        
        plt.draw()
        plt.pause(0.001)

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig('training_acc_loss.png')

class PlotAccuracy(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.title("Training Accuracy [Epoch {}]".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show();


class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.title("Training Loss [Epoch {}]".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        
        plt.legend()
        plt.show()
        
class saveModel(Callback):

    def __init__(self, save_file_weights, n=10):
        Callback.__init__(self)
        self.save_file_weights = save_file_weights
        self.n = n

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        # Testing, saving the test history and eventually saving the model weights every Nth epoch:
        if (epoch % self.n) == 9:
            print("")
            # calculate new test loss and test accuracy:
            self.model.save_weights(self.save_file_weights)
            

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

# Callback to test and save the outcoming after every 10th epoch. 
# Also saves the models parameters in case it is the best performing state:
class TestAndSaveEveryN(Callback):

    def __init__(self, x_test, y_test, save_file_weights, save_file_hist, n=10):
        Callback.__init__(self)
        self.x_test = x_test
        self.y_test = y_test
        self.save_file_weights = save_file_weights
        self.save_file_hist = save_file_hist
        self.n = n
        self.test_acc = []
        self.test_loss = []

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        # Testing, saving the test history and eventually saving the model weights every Nth epoch:
        if (epoch % self.n) == 9:
            print("")
            # calculate new test loss and test accuracy:
            new_loss, new_accuracy = self.model.evaluate(self.x_test, self.y_test)
            print("loss: {} - acc: {}".format(new_loss, new_accuracy))
            # if new loss is smaller than minimum of old losses: save weights to file:
            if not self.test_loss:
                print("Not saved yet --> save model")
                self.model.save_weights(self.save_file_weights)
            elif new_loss < min(self.test_loss):
                print("New loss is smaller --> save model")
                self.model.save_weights(self.save_file_weights)
            else:
                print("New loss is bigger --> dont save model")
            # append new loss and accuracy to list:
            self.test_acc.append(new_accuracy)
            self.test_loss.append(new_loss)
            # save the complete list:
            loss_hist = np.array(self.test_loss)
            acc_hist = np.array(self.test_acc)
            np_hist = np.vstack((loss_hist, acc_hist))
            np.savetxt(self.save_file_hist, np_hist, delimiter=" ")


    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return



