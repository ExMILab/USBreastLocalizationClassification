import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import cv2
from matplotlib import pyplot as plt

from Data import AutoencoderDataset, PredAutoencoderDataset, PredAutoencoderDatasetMask
from LinearAutoEncoders_128 import AutoEncoderDecoder
#from LinearAutoEncoders_512 import AutoEncoderDecoder

#Most of the parameters required for training
import argparse

main = 'V:\\Projekte\\2021_Zuza_Breast_nnU-net\\Revision_2\\SEGMENTATION_VIDEOS\\'

folders = ['002_019_20240206','002_020_20240206'] #,'002_115_20231223','002_215_20231223','002_315_20231223']  #'002_010_20230807'
models = ['ensemble', 'fold_0','fold_1','fold_2','fold_3','fold_4', 'ensemble_1','fold_01','fold_11','fold_21','fold_31','fold_41','ensemble_2','fold_02','fold_12','fold_22','fold_32','fold_42']

dataset = 'Autoencoder'
results_folder = 'Results_128x128_without_CBAM_nnUnet'

model_path = 'D:\\nnUNET\\Autoencoder\\OLD\\'
for i in range(len(folders)):
    folder = folders[i]
    for ii in range(len(models)):
        parser = argparse.ArgumentParser()
        parser.add_argument("--folder_path", type=str, default= main + '\\' + folder + '\\'+ models[ii] + '\\' + dataset + '\\images\\', help="Give the path for training images.")
        parser.add_argument("--out_path", type=str, default= main +'\\' + folder + '\\'+ models[ii] + '\\' + dataset + '\\results\\', help="Give the path for output.")
        parser.add_argument("--learned_features", type=int, default=1024, help="The length of the learned feature vector.")
        parser.add_argument("--saved_model_path", type=str, default= model_path + results_folder + '\\autoencoderdecoder_model_500.pth', help="If need to be loaded the starting model from previously save model.")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        parser.add_argument("--batch_size", type=int, default=1, help="Give the batch size for training.")
        parser.add_argument("--epochs", type=int, default=500, help="Give the max number of epochs for training.")
        parser.add_argument("--lr", type=float, default=0.001, help="Give learning rate for training.")
        parser.add_argument("--train_test_split", type=float, default=0.95, help="Give learning rate for training.")
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        args = parser.parse_args()


        if __name__ == '__main__':
            #GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            #Augmentation ?

            #load names of all images in the feature folder
            image_paths = glob.glob(args.folder_path+"\*.jpg")

            #Dataset
            testset = PredAutoencoderDataset(image_paths)
            N = len(testset)
            print ('Total images : ', N)
            print ('Total validation images : ',len(testset))

            #Dataloader
            testloader =  DataLoader(testset, batch_size= args.batch_size, shuffle = True)

            #Define model and load from the saved model
            model = AutoEncoderDecoder(args.learned_features)
            model.load_state_dict(torch.load(args.saved_model_path))

            # Move both the encoder and the decoder to the selected device
            model.to(device)
            model.eval()

            with torch.no_grad(): # No need to track the gradients
                # Define the lists to store the outputs for each batch
                conc_out = []
                conc_label = []
                all_features = {}

                #loop through the batch
                for j,image_batch in enumerate(testloader):

                    input = image_batch['image']
                    img_name = image_batch['name'][0]
                    print (img_name)

                    # Move tensor to the proper device
                    input = input.to(device)
                    # Encode Decode data
                    decoded_data,pred_features = model(input)

                    # Append the network output and the original image to the lists
                    conc_out.append(decoded_data.cpu())
                    all_features[img_name] = pred_features.cpu().detach().numpy()

                    #Learned features
                    #print (pred_features)

                    #saving the output images
                    outpath = args.out_path
                    display_output = decoded_data[0].cpu().detach().numpy()[0] *255
                    cv2.imwrite(os.path.join(outpath,'pred_'+ img_name +'.jpg'), display_output)

            
            cols = ['val_'+str(i) for i in range(args.learned_features +1  )]
            cols[0] = 'name'
            df = pd.DataFrame( columns=  cols )

            for i,key in enumerate(all_features.keys()):
                df.loc[i] =  [key] + list(all_features[key][0])

            print (df)
            df.to_excel( os.path.join(args.out_path,"Test_video_AE_original.xlsx"),
                sheet_name='Original', index= False) 




        
            
            
