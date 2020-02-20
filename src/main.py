""" Siamese network on MNIST data """

import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.datapairs import BalancedPairs, RandomPairs, Triplets
from src.trainer import fit
from src.networks import FeatureExtractor, SiameseNetBin, ClassificationNet, SiameseNetCont, TripletNet
from src.osnet import OSBlock, OSNet
from src.losses import BalancedBCELoss, ContrastiveLoss, TripletLoss
from src.metrics import BinAccumulatedAccuracy, AccumulatedAccuracy, ContrastiveAccuracy, AverageNonzeroTriplets

cudnn.benchmark = True
do_learn = True
batch_size = 128
lr = 0.01
n_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

mnist_training = datasets.MNIST('data/', train=True, download=True, transform=transform)
mnist_testing = datasets.MNIST('data/', train=False, download=True, transform=transform)

def fit_cross_entropy():
    # feature_extractor = FeatureExtractor()
    feature_extractor = OSNet(blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[16, 64, 96, 128])
    model = ClassificationNet(feature_extractor, feature_dim=512, n_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(mnist_training, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(mnist_testing, batch_size=batch_size, shuffle=False, **kwargs)

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=torch.nn.NLLLoss(),
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        log_interval=10,
        metrics=[AccumulatedAccuracy()],
    )

    torch.save(model.state_dict(), 'models/cross_entropy_osnet.pt')


def fit_contrastive_loss():
    feature_extractor = FeatureExtractor()
    model = SiameseNetCont(feature_extractor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_data = RandomPairs(mnist_training)
    testing_data = RandomPairs(mnist_testing)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, **kwargs)

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=ContrastiveLoss(margin=1.),
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        log_interval=10,
        metrics=[ContrastiveAccuracy()],
    )

    torch.save(model.state_dict(), 'models/contrastive_loss.pt')


def fit_bce_loss():
    feature_extractor = FeatureExtractor()
    model = SiameseNetBin(feature_extractor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_data = BalancedPairs(mnist_training)
    testing_data = BalancedPairs(mnist_testing)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, **kwargs)

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=BalancedBCELoss(),
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        log_interval=10,
        metrics=[BinAccumulatedAccuracy()],
    )

    torch.save(model.state_dict(), 'models/bce_loss.pt')


def fit_triplet_loss():
    feature_extractor = FeatureExtractor()
    model = TripletNet(feature_extractor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_data = Triplets(mnist_training)
    testing_data = Triplets(mnist_testing)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, **kwargs)

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=TripletLoss(margin=1.),
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        log_interval=10,
        metrics=[AverageNonzeroTriplets()],
    )

    torch.save(model.state_dict(), 'models/triplet_loss.pt')


if __name__ == '__main__':
    fit_triplet_loss()



