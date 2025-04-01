# https://github.com/BayesWatch/cinic-10/blob/master/notebooks/cifar-extraction.ipynb

import os
import glob
import numpy as np
from shutil import copyfile


def extract_cifar(cinic_directory, cifar_directory):

    symlink = False    # If this is false the files are copied instead
    combine_train_valid = False    # If this is true, the train and valid sets are ALSO combined
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sets = ['train', 'valid', 'test']
    if not os.path.exists(cifar_directory):
        os.makedirs(cifar_directory)
    if not os.path.exists(cifar_directory + '/train'):
        os.makedirs(cifar_directory + '/train')
    if not os.path.exists(cifar_directory + '/test'):
        os.makedirs(cifar_directory + '/test')
        
    for c in classes:
        if not os.path.exists('{}/train/{}'.format(cifar_directory, c)):
            os.makedirs('{}/train/{}'.format(cifar_directory, c))
        if not os.path.exists('{}/test/{}'.format(cifar_directory, c)):
            os.makedirs('{}/test/{}'.format(cifar_directory, c))
        if not combine_train_valid:
            if not os.path.exists('{}/valid/{}'.format(cifar_directory, c)):
                os.makedirs('{}/valid/{}'.format(cifar_directory, c))

    for s in sets:
        for c in classes:
            source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
            filenames = glob.glob('{}/*.png'.format(source_directory))
            for fn in filenames:
                dest_fn = os.path.basename(fn)
                if (s == 'train' or s == 'valid') and combine_train_valid and 'cifar' in fn.split('/')[-1]:
                    dest_fn = '{}/train/{}/{}'.format(cifar_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                    
                elif (s == 'train') and 'cifar' in fn.split('/')[-1]:
                    dest_fn = '{}/train/{}/{}'.format(cifar_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                        
                elif (s == 'valid') and 'cifar' in fn.split('/')[-1]:
                    dest_fn = '{}/valid/{}/{}'.format(cifar_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                        
                elif s == 'test' and 'cifar' in fn.split('/')[-1]:
                    dest_fn = '{}/test/{}/{}'.format(cifar_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)