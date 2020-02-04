""" Siamese network on MNIST """

import os
import random
import glob
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True

do_learn = True
batch_size = 256
lr = 0.01
num_epochs = 5
weight_decay = 0.0001


class BalancedPairs(Dataset):
    """
    Dataset that on each iteration provides a random triplet of
    MNIST images. The first two images have the same class and
    the third image has a different class.
    """

    def __init__(self, root='data', n_pairs=500, train=True, transform=None, target_transform=None,
                 download_mnist=True, save_pairs=True):
        self.root = root
        self.n_pairs = n_pairs
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.download = download_mnist

        if (self.train and glob.glob('*/*/train*.pt')) or (not self.train and glob.glob('*/*/test*.pt')):
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: add batch normalization
        self.conv1 = nn.Conv2d(1, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(2304, 512)
        self.linear2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []
        for i in range(2):  # sharing weights
            x = data[i]
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            res.append(F.relu(x))

        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        return res


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def train(model, device, train_loader, epoch, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device).float()

        optimizer.zero_grad()
        output_positive = model(data[:2])             # data[0, 1]
        output_negative = model(data[0:3:2])          # data[0, 2]

        target = target.type(torch.LongTensor).to(device)
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])

        # TODO: implement contrastive/triplet loss
        loss_positive = F.cross_entropy(output_positive, target_positive)
        loss_negative = F.cross_entropy(output_negative, target_negative)
        # binary verification loss
        loss = loss_positive + loss_negative
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx * batch_size / len(train_loader.dataset),
                loss.item()))


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device).float()

            output_positive = model(data[:2])
            output_negative = model(data[0:3:2])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = F.cross_entropy(output_positive, target_positive)
            loss_negative = F.cross_entropy(output_negative, target_negative)

            # identity loss + triplet loss
            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))


def oneshot(model, device, data):
    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device).float()

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    model = Net().to(device)

    if do_learn:  # training mode
        # load data
        training_data = BalancedPairs('data', n_pairs=2000, train=True, transform=trans, save_pairs=True)
        testing_data = BalancedPairs('data', n_pairs=500, train=False, transform=trans, save_pairs=True)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            train(model, device, train_loader, epoch, optimizer)
            test(model, device, test_loader)

        # save trained model
        torch.save(model.state_dict(), 'models/siamese.pt')

    else:  # prediction mode
        testing_data = BalancedPairs('data', n_pairs=100, train=False, transform=trans, save_pairs=True)
        prediction_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)
        model.load_state_dict(torch.load('models/siamese.pt'))
        data = []
        data.extend(next(iter(prediction_loader))[0][:3:2])
        same = oneshot(model, device, data)
        if same > 0:
            print('These two images are of the same number')
        else:
            print('These two images are not of the same number')


if __name__ == '__main__':
    main()



