import keras
import numpy as np

class Cbk(keras.callbacks.Callback):

    def __init__(self, model, label, RUN):
        self.model_to_save = model
        self.label = label
        self.run = RUN
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        # save model
        self.model_to_save.save('/output/{}_label_{}_epoch_{}_val-loss_{}.h5'.format(str(self.run), self.label, str(epoch), str(logs["val_loss"])))

    def on_train_end(self, logs={}):
        np.save('/output/{}_label_{}_loss.npy'.format(str(self.run), self.label), self.losses)
        np.save('/output/{}_label_{}_val_loss.npy'.format(str(self.run), self.label), self.val_losses)
