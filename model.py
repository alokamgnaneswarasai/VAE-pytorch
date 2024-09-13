import torch
import torch.nn as nn
import torch.nn.functional as F

# Lets write ConvVAE model

class VAE(nn.Module):
    
    def __init__(self,z_dim=128,img_size=64):
        super(VAE,self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
                                     nn.ReLU(),
                                     nn.Flatten())
        
        self.fc_mu = nn.Linear(256*(img_size//8)*(img_size//8),z_dim)
        self.fc_logvar = nn.Linear(256*(img_size//8)*(img_size//8),z_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(z_dim,256*(img_size//8)*(img_size//8))
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1),
                                    nn.Tanh())
        
        
    def encode(self,x):
        h = self.encoder(x)
        mu,logvar = self.fc_mu(h),self.fc_logvar(h)
        return mu,logvar
    
    def decode(self,z):
        h = self.fc_decode(z)
        h = h.view(-1,256,8,8)
        return self.decoder(h)
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        
     
        mu,logvar = self.encode(x)
       
        z = self.reparameterize(mu,logvar)
        return self.decode(z),mu,logvar
    
    
def loss_function(recon_x,x,mu,logvar):
    
    BCE = F.mse_loss(recon_x,x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD