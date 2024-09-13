import torch
import matplotlib.pyplot as plt
from model import VAE
from torchvision import transforms
from dataloader import get_celeba_loader

def show_images(images,title):
    
    images = images.detach().cpu().numpy().transpose(0,2,3,1)
    # images = (images+1)/2
    
    fig,ax = plt.subplots(1,len(images),figsize=(20,20))
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].axis('off')
    fig.suptitle(title)
    plt.show()
    
    # save images using title 
    plt.savefig(title+'.png')
    plt.close()
    
    
def test_vae(model,dataloader,device):
    model.eval()
    with torch.no_grad():
        for i,(x) in enumerate(dataloader):
            x = x.to(device)
            recon_x,_,_= model(x)
            show_images(x,'Original Images')
            show_images(recon_x,'Reconstructed Images')
            
            # Generate new images from random noise
            z = torch.randn(8,128).to(device)
            gen_images = model.decode(z)
            show_images(gen_images,'Generated Images')
            break
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    img_size = 64
    z_dim = 128
    data_dir = 'datafolder/img_align_celeba'
    
    dataloader = get_celeba_loader(batch_size,img_size,data_dir)
    model = VAE(z_dim,img_size).to(device)
    model.load_state_dict(torch.load('vae2.pth',weights_only=True))
    
    test_vae(model,dataloader,device)
        
        