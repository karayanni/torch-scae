from PIL import Image, ImageOps
import pathlib
import glob
import os
import cv2
import numpy as np 

desired_size = 256
home_path = str(pathlib.Path.home())
im_pth = "C:\\GitHub\\torch-scae\\torch_scae\\data\\boneage\\train_unproccessed\\training_items\\"
im_resized_pth = "C:\\GitHub\\torch-scae\\torch_scae\\data\\boneage\\train\\train0\\"
file_ext =".png"

#Creating a new directory if does not exist
try:
    if not os.path.isdir(im_resized_pth):
        os.makedirs(im_resized_pth, 777, exist_ok=True)
except OSError:
    print ("Creation of the directory %s failed" % im_resized_pth)
else:
    print ("Successfully created the directory %s " % im_resized_pth)

test_filelist = glob.glob(im_pth + '*' + file_ext) # returns a list of all the pngs in the folder - not in order

for i in range(len(test_filelist)):
    filename = test_filelist[i].split('\\')[-1]
    # new_im.save(os.path.join(im_resized_pth, filename+ "_after"), "PNG") 
    image = cv2.imread(test_filelist[i]) 
    # Resizing the image for compatibility 
    image = cv2.resize(image, (256, 256)) 
    
    # The initial processing of the image 
    # image = cv2.medianBlur(image, 3) 
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # The declaration of CLAHE  
    # clipLimit -> Threshold for contrast limiting 
    clahe = cv2.createCLAHE(clipLimit = 1) 
    final_img = clahe.apply(image_bw) + 30

    Image.fromarray(final_img).save(os.path.join(im_resized_pth, filename), "PNG") 
#new_im.show()