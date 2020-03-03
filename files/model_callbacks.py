import keras
import numpy as np

class model_callbacks(keras.callbacks.Callback):

    def __init__(self, model, RUN, dir_name):
        self.model_to_save = model
        self.run = RUN
        self.dir_name = dir_name
        self.losses = []
        self.val_losses = []
        self.best_val_loss = 1

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        # save model
        if val_loss < self.best_val_loss:
            self.model_to_save.save(self.dir_name + '/{}.h5'.format(self.run))
            print('model saved.')
            self.best_val_loss = val_loss

        # save logs (overwrite)
        np.save(self.dir_name + '/{}_loss.npy'.format(self.run), self.losses)
        np.save(self.dir_name + '/{}_val_loss.npy'.format(self.run), self.val_losses)
        print('log saved.')
