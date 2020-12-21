import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils
import data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/test',
                    help="Directory containing the dataset")
parser.add_argument('--model', default='alexnet',
                    help="The model you want to test")
parser.add_argument('--weights', required=True,
                    help="The weights file you want to test")
parser.add_argument('--batch_size', default=256,
                    help="batch size")
parser.add_argument('--gpu', action='store_true', default='False',
                    help="GPU available")


def evaluate(model, loss_fn, dataloader):
    """ Evaluate the model on `num_steps` batches
    Args:
        model : (torch.nn.Module) model
        dataloader : (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        num_steps : (int) # of batches to train on, each of size args.batch_size
    """

    # set model to test mode
    model.eval()

    model_dir = './results/' + model_name

    total_loss = 0.0
    total_correct = 0.0

    for i, (test_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        if args.gpu:
            test_batch, labels_batch = test_batch.cuda(), labels_batch.cuda()

        # convert to torch Variable
        test_batch, labels_batch = Variable(test_batch), Variable(labels_batch)

        # compute model output and loss
        output_batch = model(test_batch)
        loss = loss_fn(output_batch, labels_batch)

        total_loss += loss.item()
        acc = utils.accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
        total_correct += acc

    print("Loss:{:.4f}\t Test Accuracy:{:.4f}".format(
        total_loss/len(dataloader),
        100 * total_correct / len(dataloader)
    ))

if __name__ == '__main__':

    # Load the parameters from parser
    args = parser.parse_args()

    model_name = args.model
    weights_path = args.weights
    batch_size = args.batch_size

    logging.info("Loading the test dataset...")

    # fetch train dataloader
    test_dataloader = data_loader.test_data_loader()

    logging.info("- done.")

    # Define the model and optimizer
    model = utils.get_network(args)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict'])

    # fetch loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    logging.info("Starting Test ...")
    evaluate(model, loss_fn, test_dataloader)