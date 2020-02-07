
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image


class BalancedPairs(Dataset):
    """
    Dataset that on each iteration provides a random triplet of
    MNIST images. The first two images have the same class and
    the third image has a different class.
    """

    def __init__(self, dataset, n_pairs=500):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.images = self.dataset.train_data
            self.labels = self.dataset.train_labels
        else:
            self.images = self.dataset.test_data
            self.labels = self.dataset.test_labels

        # split images/labels dataset along the 10 classes
        train_labels_class = []
        train_images_class = []
        for i in range(10):
            indices = torch.squeeze((self.labels == i).nonzero())
            train_labels_class.append(torch.index_select(self.labels, 0, indices))
            train_images_class.append(torch.index_select(self.images, 0, indices))

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
                    train_images_class[i][j % class_len[i]],  # get an image from class i
                    train_images_class[i][(j + rnd_dist) % class_len[i]],  # get a different image from class i
                    train_images_class[rnd_cls][j % class_len[rnd_cls]]  # get an image from a different class
                ]))
                self.labels.append([1, 0])

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, item):
        images, labels = self.images[item], self.labels[item]
        images_list = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i].numpy(), mode='L')
            if self.transform:
                img = self.transform(img)
            images_list.append(img)
        return tuple(images_list), labels

    def __len__(self):
        return len(self.images)


class RandomPairs(Dataset):
    """
    Train: For each sample in the input dataset creates randomly a positive or a negative pair.
    Test: Creates fixed pairs for testing.
    The input dataset should be a pytorch Dataset class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - {self.test_labels[i].item()})
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - {label1}))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)


class Triplets(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - {self.test_labels[i].item()})
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - {label1}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        super(BalancedBatchSampler, self).__init__()
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

# get MNIST images and labels
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#
# mnist_data = datasets.MNIST(
#     '../data',
#     train=False,
#     download=True,
#     transform=transform,
# )
#
# data = Triplets(mnist_data)
# print(len(data))
