"""
Contains functions to troubleshoot the models.
"""
from matplotlib import pyplot as plt
import keras.callbacks as cbks


def plot_eval_metrics(history):
    """
    Plots the loss and accuracy of the model as it was trained. Example usage:

    history = model.fit(....)
    plot_eval_metrics(history)

    :param history: A keras.callbacks.History object
    :return: nothing
    """

    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


class CustomMetrics(cbks.Callback):

    def __init__(self, metrics):
        super(CustomMetrics, self).__init__()
        self.metrics = {}

        for metric in metrics:
            self.metrics[metric] = -1

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k in self.metrics:
                self.metrics[k] = logs[k]
