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
parser.add_argument('--data_dir', default='./data/train',
                    help="Directory containing the dataset")
parser.add_argument('--model', type=str, required=True,
                    help="The model you want to train")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate")
parser.add_argument('--epoch', type=int, default=50,
                    help="Total training epochs")
parser.add_argument('--batch_size', type=int, default=256,
                    help="batch size")
parser.add_argument('--gpu', action='store_true', default='False',
                    help="GPU available")


def train(model, optimizer, loss_fn, dataloader):
    """ Train the model on `num_steps` batches
    Args:
        model : (torch.nn.Module) model
        optimizer : (torch.optim) optimizer for parameters of model
        loss_fn : (string) a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader : (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        num_steps : (int) # of batches to train on, each of size args.batch_size
    """

    # set model to training mode
    model.train()

    model_dir = './results/' + model_name
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0.0

        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if args.gpu:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

            # convert to torch Variable
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)


            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            epoch_loss += loss.item()
            acc = utils.accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
            epoch_correct += acc

#             print("Epoch [{}]\t Batch [{}/{}]\t Loss:{:.4f}\t Accuracy:{:.4f}".format(epoch+1, i, len(dataloader), loss.item(), acc))

        print("Epoch [{}/{}]\t Loss:{:.4f}\t Accuracy:{:.4f}%".format(
            epoch + 1,
            epochs,
            epoch_loss/len(dataloader),
            100 * epoch_correct / len(dataloader)
        ))

        is_best = acc >= best_acc
        if is_best:
            logging.info("- Found new best accuracy")
            best_acc = acc

        utils.save_checkpoints(
            {'epoch': i + 1,
             'state_dict': model.state_dict(),
             'optim_dict': optimizer.state_dict()},
            is_best=is_best,
            checkpoint=model_dir
        )

if __name__ == '__main__':

    # Load the parameters from parser
    args = parser.parse_args()

    model_name = args.model
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batch_size

    logging.info("Loading the training dataset...")

    # fetch train dataloader
    train_dataloader = data_loader.train_data_loader()

    logging.info("- done.")

    # Define the model and optimizer
    model = utils.get_network(args)
    optimizer = utils.get_optimizer(model_name, model, lr)

    # fetch loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    logging.info("Starting training for {} epoch(s).".format(epochs))
    train(model, optimizer, loss_fn, train_dataloader)