"""
This script losses prints the losses for the training of the trainSkeletonMorphing.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main).
"""

import numpy as np


def print_losses(epoch, iteration, iter_per_epoch, losses, print_keys=False):
    """
    Print the training losses for a specific epoch and iteration.

    Parameters:
    - epoch: Current epoch number.
    - iter: Current iteration number.
    - iter_per_epoch: Total number of iterations per epoch.
    - losses: Dictionary containing different loss values.
    - print_keys: Flag to print column headers if True.

    Example Usage:
    print_losses(1, 100, 1000, {'loss': 0.123, 'accuracy': 0.95}, print_keys=True)
    """

    # Print column headers if requested
    if print_keys:
        header_str = 'epoch %d\t\t\tloss\t' % epoch

        # Loop through keys in losses dictionary to print column headers
        for key, value in losses.items():
            if key != 'loss':
                # Format column headers with padding for better alignment
                if len(key) < 5:
                    key_str = key + ' ' * (5 - len(key))
                    header_str += '\t\t%s' % key_str
                else:
                    header_str += '\t\t%s' % (key[0:5])

        print(header_str)

    # Format and print the loss values
    loss_str = '%05d/%05d: \t%.4f\t' % (iteration, iter_per_epoch, np.mean(losses['loss']))

    # Loop through keys in losses dictionary to print loss values
    for key, value in losses.items():
        if key != 'loss':
            loss_str += '\t\t%.4f' % (np.mean(value))

    print(loss_str)