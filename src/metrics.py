import numpy as np
import torch


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracy(Metric):
    """
    Works with classification model.
    """

    def __init__(self):
        super(AccumulatedAccuracy, self).__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

class ContrastiveAccuracy(Metric):
    """
    Works with contrastive model.
    """

    def __init__(self):
        super(ContrastiveAccuracy, self).__init__()
        self.correct = 0
        self.total = 0

    # TODO: use Euclidean distance on normalized features
    def __call__(self, outputs, target, loss):
        distance = torch.nn.CosineSimilarity(dim=1)
        pred = distance(outputs[0], outputs[1]) > 0
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class BinAccumulatedAccuracy(Metric):
    """ Works with binary siamese network """

    def __init__(self):
        super(BinAccumulatedAccuracy, self).__init__()
        self.correct = 0
        self.total = 0

    # TODO: how is accuracy computed ?
    def __call__(self, outputs, target, loss):
        target = target[0] if type(target) in (tuple, list) else target
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])
        acc_positive = torch.sum(torch.argmax(outputs[0], dim=1) == target_positive).cpu()
        acc_negative = torch.sum(torch.argmax(outputs[1], dim=1) == target_negative).cpu()
        self.correct += acc_positive + acc_negative
        self.total += target_positive.size(0) + target_negative.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


# TODO: compute accuracy
class AverageNonzeroTriplets(Metric):
    """
    Counts average number of nonzero triplets found in minibatches.
    """

    def __init__(self):
        super(AverageNonzeroTriplets, self).__init__()
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss.item())
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return 1 - np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
