import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE,loss_function
from dataloader import get_celeba_loader

def train_vae(epochs,dataloader,model,optimizer,device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i,(x) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            recon_x,mu,logvar = model(x)
            loss = loss_function(recon_x,x,mu,logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        print('Epoch: {}, Loss: {}'.format(epoch+1,total_loss/len(dataloader)))
                
                
                
        
       
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4096
    img_size = 64
    z_dim = 128
    data_dir = 'datafolder/img_align_celeba'
    epochs = 20
    
    print('Data loading...')
    dataloader = get_celeba_loader(batch_size,img_size,data_dir)
    print('Data loaded...')
    model = VAE(z_dim,img_size).to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    train_vae(epochs,dataloader,model,optimizer,device)
    
    # save model
    torch.save(model.state_dict(),'vae1.pth')