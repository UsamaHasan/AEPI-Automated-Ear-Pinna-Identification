from __future__ import print_function , division
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np

def loader(root_dir):
    """Dataset Loader"""
    parent_path_list = os.listdir(root_dir)
    minimun_height = 224
    minimum_width = 224  
    flag = 0
    count = 0
    image_list = []
    label_list = []
    number_of_classes = 5
    for parent_path in parent_path_list:
        if count < number_of_classes:
            parent_path = os.path.join(root_dir, parent_path)
            child_path_list = os.listdir(parent_path)
            for child_path in child_path_list:
                image_path = os.path.join(parent_path,child_path)
                image = io.imread(image_path)
                

                """We have to preprocess our images according to our model requirements,
                since the dataset contains images of variable dimension, we need to upscale/downscale 
                them to a fixed dimension. Here were consider using vgg-16 as our backbone model
                so the input dimension has to be (100*100*3).
                For that purpose, images above 100 will be downscaled and images below will be zero
                padded.
                After the end of first block images will be either converted to 100,100*3."""

                """First Block."""

                if(image.shape[0] >= minimun_height or image.shape[1] >= minimum_width):
                    #Downsize the image dim.
                    rescaled_image = resize(image,(minimum_width,minimun_height))
                #Zero Padding.

                else:
                    #For Height.
                    if(image.shape[0] % 2 == 0 ):
                        pad_top_bottom = int((minimun_height - image.shape[0])/2)
                    else:
                        pad_top_bottom = int((minimun_height + 1 - image.shape[0])/2)
        
                    #For width.
                    if (image.shape[1] % 2 == 0 ):
                        #if Even 
                        pad_left_right = int((minimum_width - image.shape[1])/2)
                    else:
                        pad_left_right = int((minimum_width +1 - image.shape[1])/2)
                        
                    #loop for the three channels
                    i = 0

                    height = 2*pad_top_bottom + image.shape[0]
                    width  = 2*pad_left_right + image.shape[1]

                    padded_image = np.zeros((height,width,3) , 'uint8')

                    while(i<3):
                        arr = image[:,:,i : i+1]
                        arr = arr.reshape((image.shape[0],image.shape[1]))
                        padded_image[:,:,i] = np.pad(arr, ((pad_top_bottom,pad_top_bottom),(pad_left_right,pad_left_right)),'constant')
                        i+=1
                    
                    del arr
                        
                    #If height/width is even odd, then the resulatant dim will be 224,223
                    if(padded_image.shape[0] == minimum_width + 1 or padded_image.shape[1] == minimun_height +1):
                        rescaled_image = resize(padded_image,(minimum_width,minimun_height,3))
                    else:
                        rescaled_image = padded_image
                    
                    #To vizualize uncomment this line.
                    #plt.imshow(rescaled_image)
                    #plt.show()

                """End of First Block."""
                    
                #The first 3 literals of the image file contains the label/class of image.
                
                label = child_path[:3]
                try:
                    label = int(label)
                    print(label)
                    image_list.append(rescaled_image)
                    label_list.append(label)
                except:
                    print('Exception')
                del rescaled_image
        count+=1        
    image_array = np.array(image_list)
    label_array = np.array(label_list)

    print(f'Dataset Dimesion: {image_array.shape}')
    print(f'Labels Dimension: {label_array.shape}')
    np.save('images',image_array)
    np.save('labels',label_array)


if __name__ == '__main__':
    #Enter path of dataset here.
    loader('/home/usamahasan/Dataset/EarVN1.0 dataset/Images/')