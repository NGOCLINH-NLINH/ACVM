import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCVAE(nn.Module):
    def __init__(self, embed_dim=384, latent_dim=128, num_classes=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.label_emb = nn.Embedding(num_classes, 64)

        self.enc1 = nn.Linear(embed_dim + 64, 512)
        self.enc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.dec1 = nn.Linear(latent_dim + 64, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, embed_dim)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.leaky_relu(self.enc1(inputs), 0.2)
        h = F.leaky_relu(self.enc2(h), 0.2)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = F.leaky_relu(self.dec1(inputs), 0.2)
        h = F.leaky_relu(self.dec2(h), 0.2)
        recon = self.dec3(h)
        return F.normalize(recon, p=2, dim=1)

    def forward(self, x, labels):
        c = self.label_emb(labels)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def generate(self, labels, device):
        c = self.label_emb(labels)
        z = torch.randn(labels.size(0), self.latent_dim).to(device)
        return self.decode(z, c)


def cvae_loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (MSE + KLD) / x.size(0)