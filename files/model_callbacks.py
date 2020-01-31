import keras

class Cbk(keras.callbacks.Callback):

    def __init__(self, model, label, RUN):
        self.model_to_save = model
        self.label = label
        self.run = RUN

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        # np.save("siko.npy", self.losses)

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('/output/{}_label_{}_epoch_{}_val-loss_{}.h5'.format(str(self.run), self.label, str(epoch), str(logs["val_loss"])))
