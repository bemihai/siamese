""" Siamese network on MNIST data """

import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.datapairs import BalancedPairs, RandomPairs, Triplets
from src.trainer import fit
from src.networks import FeatureExtractor, SiameseNetBin, ClassificationNet, SiameseNetCont, SiameseNetTrip
from src.losses import BalancedBCELoss, ContrastiveLoss, TripletLoss
from src.metrics import BinAccumulatedAccuracy, AccumulatedAccuracy

cudnn.benchmark = True
do_learn = True
batch_size = 256
lr = 0.01
n_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
feature_extractor = FeatureExtractor()
# model = SiameseNetBin(feature_extractor).to(device)  # loss_fn=BalancedBCELoss(),
# model = ClassificationNet(feature_extractor, n_classes=10).to(device)  # loss_fn=torch.nn.NLLLoss(),
# model = SiameseNetCont(feature_extractor).to(device)  # loss_fn=ContrastiveLoss(margin=1.)
model = SiameseNetTrip(feature_extractor).to(device)  # loss_fn=TripletLoss(margin=1.)
optimizer = optim.Adam(model.parameters(), lr=lr)

mnist_training = datasets.MNIST('data/', train=True, download=True, transform=transform)
mnist_testing = datasets.MNIST('data/', train=False, download=True, transform=transform)

def main():

    training_data = Triplets(mnist_training)
    testing_data = Triplets(mnist_training)

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
        metrics=[],
    )

    # save trained model
    # torch.save(model.state_dict(), '../models/siamese.pt')


if __name__ == '__main__':
    main()



