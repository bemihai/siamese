import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Takes embeddings of two samples and a target label = 1 if samples are from the same class (positive pair)
    and label = 0 otherwise (negative pair).
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss function.
    Takes embeddings of an anchor sample, a positive sample and a negative sample.
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if average else losses.sum()


class BalancedBCELoss(nn.Module):
    """
    Balanced binary cross entropy loss function.
    Takes differential embeddings of a positive pair and a negative pair.
    The differential embedding of a pair is the squared distance of the corresponding embeddings.
    """
    
    def __init__(self):
        super(BalancedBCELoss, self).__init__()
    
    def forward(self, positive, negative, target, average=True):
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])
        loss_positive = F.cross_entropy(positive, target_positive)
        loss_negative = F.cross_entropy(negative, target_negative)
        losses = loss_positive + loss_negative
        return losses.mean() if average else losses.sum()
    

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss function.
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs.
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss function.
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets.
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
