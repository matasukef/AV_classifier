import numpy as np

def preprocess(x):
    x = x[:, :, ::-1]
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68

    return x


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, 'w') as f:
        f.write('epoch\tloss\tacc\tval_loss\tval_acc\n\n')
        for i in range(nb_epoch):
            f.write('{0}\t{1}\t{2}\t{3}\t{4}'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))
