import os
import random
import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class BalancedPairs(Dataset):
    """
    Dataset that on each iteration provides a random triplet of
    MNIST images. The first two images have the same class and
    the third image has a different class.
    """

    def __init__(self, root='../data', n_pairs=500, train=True, transform=None, target_transform=None,
                 download_mnist=True, save_pairs=True):
        self.root = root
        self.n_pairs = n_pairs
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.download = download_mnist

        if (self.train and glob.glob('../*/*/train*.pt')) or (not self.train and glob.glob('../*/*/test*.pt')):
            self.load_pairs()
        else:
            # get MNIST images and labels
            mnist_data = datasets.MNIST(
                self.root,
                train=self.train,
                download=self.download,
                transform=self.transform,
                target_transform=self.target_transform
            )
            if self.train:
                self.raw_images = mnist_data.train_data.unsqueeze(1)
                self.raw_labels = mnist_data.train_labels
            else:
                self.raw_images = mnist_data.test_data.unsqueeze(1)
                self.raw_labels = mnist_data.test_labels

            # split images/labels dataset along the 10 classes
            train_labels_class = []
            train_images_class = []
            for i in range(10):
                indices = torch.squeeze((self.raw_labels == i).nonzero())
                train_labels_class.append(torch.index_select(self.raw_labels, 0, indices))
                train_images_class.append(torch.index_select(self.raw_images, 0, indices))

            # generate balanced pairs (1, 1, 0)
            self.images = []
            self.labels = []
            class_len = [x.shape[0] for x in train_labels_class]
            for i in range(10):
                for j in range(n_pairs):  # number of pairs to be created for each class
                    rnd_cls = random.randint(0, 8)  # choose randomly a different class
                    if rnd_cls >= i:
                        rnd_cls += 1
                    # choose randomly a distance between 0 and 100
                    rnd_dist = random.randint(0, 100)

                    self.images.append(torch.stack([
                        train_images_class[i][j % class_len[i]],                # get an image from class i
                        train_images_class[i][(j + rnd_dist) % class_len[i]],   # get a different image from class i
                        train_images_class[rnd_cls][j % class_len[rnd_cls]]     # get an image from a different class
                    ]))
                    self.labels.append([1, 0])

            self.images = torch.stack(self.images)
            self.labels = torch.tensor(self.labels)

            if save_pairs:
                self.save_pairs()

    # save pairs to file
    def save_pairs(self):
        if self.train:
            torch.save(self.images, os.path.join(self.root, 'MNIST', 'train_pairs.pt'))
            torch.save(self.labels, os.path.join(self.root, 'MNIST', 'train_labels.pt'))
            print('MNIST training pairs saved to {}.'.format(os.path.join(self.root, 'MNIST')))
        else:
            torch.save(self.images, os.path.join(self.root, 'MNIST', 'test_pairs.pt'))
            torch.save(self.labels, os.path.join(self.root, 'MNIST', 'test_labels.pt'))
            print('MNIST testing pairs saved to {}.'.format(os.path.join(self.root, 'MNIST')))

    # load pairs from file
    def load_pairs(self):
        if self.train:
            self.images = torch.load(os.path.join(self.root, 'MNIST', 'train_pairs.pt'))
            self.labels = torch.load(os.path.join(self.root, 'MNIST', 'train_labels.pt'))
            print('Loaded training MNIST pairs from files.')
        else:
            self.images = torch.load(os.path.join(self.root, 'MNIST', 'test_pairs.pt'))
            self.labels = torch.load(os.path.join(self.root, 'MNIST', 'test_labels.pt'))
            print('Loaded testing MNIST pairs from files.')

        print('Delete files from {} if you want to generate other pairs.'.format(
            os.path.join(self.root, 'MNIST')
        ))

    def __getitem__(self, item):
        images, labels = self.images[item], self.labels[item]
        images_list = []
        for i in range(images.shape[0]):
            images_list.append(images[i])
        return images_list, labels

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of triplets: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
