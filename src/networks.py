""" Network architectures """

import torch
from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """ Basic convnet for feature extraction """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.BatchNorm2d(64), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512), nn.BatchNorm1d(512), nn.PReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

    def get_features(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    """
    Baseline classification net: add a fully-connected layer with the number of classes and
    train the feature extractor for classification with softmax and cross-entropy.
    """
    def __init__(self, feature_extractor, n_classes):
        super(ClassificationNet, self).__init__()
        self.extractor = feature_extractor
        self.n_classes = n_classes
        self.activation = nn.PReLU()
        self.linear = nn.Linear(2, n_classes)

    def forward(self, x):
        out = self.extractor(x)
        out = self.activation(out)
        out = self.linear(out)
        out = F.log_softmax(out, dim=-1)
        return out,

    # extract 2-dim features from penultimate layer
    def get_features(self, x):
        self.activation(self.extractor(x))


class SiameseNetCont(nn.Module):
    """
     Contrastive siamese net: takes a pair of images and trains the feature extractor to minimize
     the contrastive loss function.
    """

    def __init__(self, feature_extractor):
        super(SiameseNetCont, self).__init__()
        self.extractor = feature_extractor

    def forward(self, x1, x2):
        out_1 = self.extractor(x1)
        out_2 = self.extractor(x2)
        return out_1, out_2

    def get_features(self, x):
        return self.extractor(x)


class SiameseNetBin(nn.Module):
    """
     Binary siamese net: takes an image triplet (anchor, positive, negative) and
     trains the feature extractor to minimize the balanced binary cross-entropy loss function.
    """

    def __init__(self, feature_extractor):
        super(SiameseNetBin, self).__init__()
        self.extractor = feature_extractor
        self.activation = nn.PReLU()
        self.linear = nn.Linear(2, 2)

    def forward(self, x1, x2, x3):
        out_1 = self.activation(self.extractor(x1))
        out_2 = self.activation(self.extractor(x2))
        out_3 = self.activation(self.extractor(x3))
        positive = self.linear(torch.abs(out_2 - out_1))
        negative = self.linear(torch.abs(out_3 - out_1))
        return positive, negative


class SiameseNetTrip(nn.Module):
    """
    Siamese net: takes an image triplet (anchor, positive, negative) and trains the feature extractor to
    minimize the triplet loss function, i.e. the anchor is closer to the positive example than it is to
    the negative example by some margin value.
    """

    def __init__(self, feature_extractor):
        super(SiameseNetTrip, self).__init__()
        self.extractor = feature_extractor

    def forward(self, x1, x2, x3):
        out_1 = self.extractor(x1)
        out_2 = self.extractor(x2)
        out_3 = self.extractor(x3)
        return out_1, out_2, out_3

    def get_features(self, x):
        return self.extractor(x)







