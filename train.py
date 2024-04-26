import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from unet import UNet
from pconv_unet import PConvUNet
from loss import VGG16PartialLoss
from dataset import DummyCTDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # this is for mac (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# model = PConvUNet(n_class=1)
model = UNet(n_class=1)

learning_rate = 1e-4  # how much to update model after each batch/epoch
batch_size = 1  # number of samples propagated through network before parameters are updated
epochs = 10  # number of times to iterate over the dataset

loss_fn = VGG16PartialLoss(vgg_path="/zhome/clarkcs/.torch/vgg16-397923af.pth")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

dataset = DummyCTDataset("/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_100projs_center-slice.npy", 
                         "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_1000projs_center-slice.npy",
                         64)
train_dataloader = DataLoader(dataset, batch_size=batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# train_features, train_labels = next(iter(train_dataloader))
# print(train_features.dtype)
# print(train_labels.dtype)

for i in range(epochs):
    print(f"Epoch {i+1}\n---------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)

train_features, train_labels = next(iter(train_dataloader))
model.eval()
with torch.no_grad():
    pred = model(train_features)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(pred.squeeze())
    ax2.imshow(train_labels.squeeze())
    # plt.savefig("/zhome/clarkcs/Documents/repos/U-Net/slurm_training.png")
    plt.show()
