import sys
import os
import shutil

import numpy as np
import torch
import torch.optim as optim

def get_network(args):
    """ Return the given network
    Args:
        args : (argparser)
    """

    if args.model == 'alexnet':
        from models.alexnet import alexnet
        net = alexnet()
    if args.model == 'zfnet':
        from models.ZFNet import ZFNet
        net = ZFNet()
    # elif args.net == 'vgg':
    #     from models.vgg import vgg
    #     net = vgg()
    # elif ...
    #       ...

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net

def get_optimizer(model_name, model, lr):
    """ Return the optimizer
    """

    if model_name == 'alexnet':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif model_name == 'zfnet':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
    # elif args.net == 'vgg':
    #     from models.vgg import vgg
    #     net = vgg()
    # elif ...
    #       ...

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return optimizer

def accuracy(outputs, labels):
    """ Compute the accuracy
    Args:
        outputs : (nd.ndarray) prediction
        labels : (nd.ndarray) real
    Returns:
          (float) accuaracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def save_checkpoints(state, is_best, checkpoint):
    """ Saves model and training parameters at checkpoint + 'last.pth.
    If is_best==True, also saves checkpoint + 'best.pth'

    Args:
        state : (dict) contains model's state_dict
        is_best : (bool) True if it is the best model
        checkpoint : (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Direcotry doesn't exist. Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))

