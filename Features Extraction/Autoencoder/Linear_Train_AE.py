#-------------------------------------------------------------------------------------------------------------------
#With this script you can train: 512x512 (uncomment line XX) or 128x128 (uncomment line XX) convoutional autoencoder 
#-------------------------------------------------------------------------------------------------------------------
from pickle import FALSE
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import cv2
from matplotlib import pyplot as plt

from Data import AutoencoderDataset
#from LinearAutoEncoders_512 import AutoEncoderDecoder
#from LinearAutoEncoders_128 import AutoEncoderDecoder

#Most of the parameters required for trainings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, default = path_train_images, help="Give to training data") # images_all
parser.add_argument("--out_path", type=str, default = path_results, help="Give output path")

parser.add_argument("--learned_features", type=int, default=1024, help="The length of the learned feature vector.")

parser.add_argument("--batch_size", type=int, default=16, help="Give the batch size for training.")
parser.add_argument("--epochs", type=int, default=1010, help="Give the max number of epochs for training.")
parser.add_argument("--lr", type=float, default=0.001, help="Give learning rate for training.")
#parser.add_argument("--momentum", type = float, default=0.8, help ="Give momentum for training")
parser.add_argument("--train_test_split", type=float, default=0.95, help="Give train test plit ration for training.")

parser.add_argument("--show_result_per_epoch", type=bool, default=True, help="If need to be Show the training result after each epoch.(can be found in out_path)")
parser.add_argument("--load_saved_model", type=bool, default=False, help="If need to be loaded the starting model from previously save model.")
parser.add_argument("--saved_model_path", type=str, default='', help="If need to be loaded the starting model from previously save model.")
args = parser.parse_args()

#Custom Loss function: weighted mse loss <<not used>>
def weighted_mse_loss(input, target, weight):
    return torch.sum( torch.mul( torch.square(input - target), weight) )

### Training function
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()

    train_loss = []
    # Iterate the dataloader
    for data in dataloader: 
        # Move tensor to the proper device
        image_batch = data['image']
        image_batch = image_batch.to(device)
        target = data['target']
        target = target.to(device)

        weight = data['mask']
        weight = weight.to(device) 

        # Encode Decode data
        decoded_data, y = model(image_batch)
        
        # Evaluate loss
        loss = 100 * loss_fn(decoded_data,target) #weighted_mse_loss(decoded_data, target,weight)  
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch(model, device, testloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    model.eval()

    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        masks = []
        for j,data in enumerate(testloader):
            # Move tensor to the proper device
            image_batch = data['image']
            image_batch = image_batch.to(device)
            target = data['target']
            mask = data['mask']
            target = target.to(device)

            # Encode Decode data
            decoded_data,y = model(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(target.cpu())
            masks.append(mask)        
            
            if args.show_result_per_epoch:
                if j == 0:   #save the image to see what is learning
                    outpath = args.out_path
                    display_input = image_batch[j].cpu().detach().numpy()[0] *255
                    display_output = decoded_data[j].cpu().detach().numpy()[0] *255
                    msk_out = mask[j].detach().numpy()[0] *255
                    #print (display_input.shape, msk_out.shape, np.unique(msk_out) )
                    print ('Evaluating on image ', data['name'][j])
                    cv2.imwrite(os.path.join(outpath,'input.jpg'), display_input )
                    cv2.imwrite(os.path.join(outpath,'mask.jpg'), msk_out )
                    cv2.imwrite(os.path.join(outpath,'output.jpg'), display_output)

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = 100 * loss_fn(conc_out, conc_label)
    return val_loss.data

'''
Training of Autoencoder
'''
if __name__ == '__main__':
    #GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Augmentation ?

    #load names of all images in the feature folder
    image_paths = glob.glob(args.folder_path+"\*.*")

    #Dataset
    dataset = AutoencoderDataset(image_paths)
    N = len(dataset)
    print ('Total images : ', N)

    # Trainset Test set division
    trainset, testset =  random_split(dataset, [ int(np.floor(N* args.train_test_split)), N -  int(np.floor(N* args.train_test_split))]) 

    #display length of datasets
    print ('Total training images : ', len(dataset) )
    print ('Total validation images : ',len(testset))

    #Dataloader
    dataloader =  DataLoader(dataset, batch_size= args.batch_size, shuffle = True)
    testloader =  DataLoader(testset, batch_size= args.batch_size, shuffle = True)

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    model = AutoEncoderDecoder(args.learned_features)
    if args.load_saved_model:
        model.load_state_dict(torch.load(os.path.join(args.out_path,'autoencoderdecoder_model.pth')))

    #loss function
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr) # weight_decay=1e-05

    # Move both the encoder and the decoder to the selected device
    model.to(device)
    num_epochs = args.epochs

    #Store results after epochs
    Best = 0
    Best_epoch = 0
    Epochs = []
    diz_loss = {'train_loss':[],'val_loss':[]}

    #Training through epochs
    for epoch in range(num_epochs):

        train_loss = train_epoch(model,device,dataloader,loss_fn,optim)
        val_loss = test_epoch(model,device,testloader,loss_fn)
        #time.sleep(2)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            
        #storing the values of current iterations
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        Epochs.append(epoch)
            
        #taking output path from the parameters
        outpath = args.out_path

        #Plotting the curves after each epochs
        plt.plot(Epochs,diz_loss['train_loss'],label="Training loss", color="blue")
        #plt.plot(Epochs,diz_loss['val_loss'],label="Validation loss", color="red")
        if epoch == 0:   # once the annotation in the curve plot
                plt.legend()
        plt.savefig(os.path.join(outpath,'train_progress.png'))

        # Saving model after each epoch (overwrites each time)
        torch.save(model.state_dict(), os.path.join(outpath,'autoencoderdecoder_model.pth') ) 

        #Saving model in a new name after every N epochs
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(outpath,'autoencoderdecoder_model_'+ str(epoch) +'.pth') ) 

        if epoch == 150 :
            for g in optim.param_groups:
                g['lr'] = 0.001
    plt.clf()
