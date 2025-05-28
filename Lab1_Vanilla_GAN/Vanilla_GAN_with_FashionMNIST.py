import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Hyperparameters
batch_size = 128
lr = 0.0002
z_dim = 100
num_epochs = 1000


# Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])


train_data = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    
# Generator, Discriminator
G = Generator(z_dim).to(device)
D = Discriminator().to(device)


# Loss, Optimizer
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


# Noise
fixed_noise = torch.randn(64, z_dim, device=device)


# Save GIF result
def img2gif(path):
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    img_list = os.listdir(path)
    img_list = sorted(img_list, key=extract_number)
    img_list = [os.path.join(path, x) for x in img_list]
    images = [Image.open(x) for x in img_list]
    
    im = images[0]
    im.save('result.gif', save_all=True, append_images=images[1:], loop=0xff, duration=200)


# Train
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.view(-1, 28 * 28).to(device)
        batch_size = real.size(0)

        # Train Discriminator: real + fake
        noise = torch.randn(batch_size, z_dim, device=device)
        fake = G(noise)

        D_real = D(real)
        D_fake = D(fake.detach())
        loss_D = criterion(D_real, torch.ones_like(D_real)) + criterion(D_fake, torch.zeros_like(D_fake))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator: fake Discriminator
        output = D(fake)
        loss_G = criterion(output, torch.ones_like(output))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")


    # Visualize
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            
            fake_images = G(fixed_noise).reshape(-1, 1, 28, 28) * 0.5 + 0.5
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8)
            npimg = grid_img.cpu().numpy()
            
            os.makedirs("output", exist_ok=True)
            
            plt.figure(figsize=(6,6))
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.title(f"Epoch {epoch+1}")
            plt.axis('off')
            
            plt.savefig(f"output/epoch_{epoch+1:03d}.png")
            plt.close()
    
    
    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(G.state_dict(), "checkpoints/G.pth")
    torch.save(D.state_dict(), "checkpoints/D.pth")
    img2gif('output')
