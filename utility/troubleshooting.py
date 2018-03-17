"""
Contains functions to troubleshoot the models.
"""
from matplotlib import pyplot as plt


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