# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:26:52 2020

@author: zmagnuska
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:40:38 2020

@author: zmagnuska
"""

import os
import glob
import SimpleITK as sitk
import six
import logging
import numpy as np
import pandas as pd

import radiomics
from radiomics import featureextractor

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog_4.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


#main_paths = ['V:\\Projekte\\2021_Zuza_Breast_nnU-net\\Revision\\CLASSIFICATION\\RADIOMICS\\M1_GT_segment\\Train\\',
#              'V:\\Projekte\\2021_Zuza_Breast_nnU-net\\Revision\\CLASSIFICATION\\RADIOMICS\\M2_GT_bbox\\Train\\',
#              'V:\\Projekte\\2021_Zuza_Breast_nnU-net\\Revision\\CLASSIFICATION\\RADIOMICS\\M5_the_whole_image\\Train\\']

#main_paths = ['V:\\Projekte\\2021_Zuza_Breast_nnU-net\\Revision\\CLASSIFICATION\\IMAGES\\ensemble_seg\\',
main_paths = ['E:\\ensemble_seg\\']


for path in main_paths:
    img_path = path + 'images\\' #image
    os.chdir(img_path)
    images = glob.glob('*.{}'.format('png'))
    
    #mask_path = main_path + '\\'+ user + '\\'+ lesion + '\\Masks\\'
    mask_path = path + 'masks\\' #mask
    os.chdir(mask_path)
    masks = glob.glob('*.{}'.format('png'))
    
    for i in range(225,670):
        imageName = img_path + images[i]
        s_img = os.path.splitext(images[i])
        print(imageName)
        maskName = mask_path + masks[i]
        print(maskName)
        image = sitk.ReadImage(imageName)
        mask = sitk.ReadImage(maskName)
         

        #initializing the feature extractor
        settings = {}
        settings = {'label': 255} 
        settings['binWidth'] = 25
        #settings['featureClass']='firstorder'
        #settings['resampledPixelSpacing']=None
        #settings['interpolator'] = 'sitkBSpline'
        settings['kernelRadius'] = 1
        settings['maskedkernel'] = True
        settings['verboes']=True
            
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            
        extractor.enableAllImageTypes()

        result = extractor.execute(image, mask) #, voxelBased= True
        for key, val in six.iteritems(result):
            if isinstance(val,sitk.Image):
                sitk.WriteImage(val, key,'_',s_img[0],'.png', True)
                print('Stored FMs')
                    
            
            print("\t%s: %s"%(key,val)) 
            r_array = np.asarray(list(result.items()))
            my_df = pd.DataFrame(r_array)
            #s_path = main_path + '\\'+ user + '\\'+ lesion + '\\'
            s_path = path + 'features\\'
            os.chdir(s_path)
            name_my_df = 'FMs_' + s_img[0] + '.csv'         
            my_df.to_csv(name_my_df, index = False)        