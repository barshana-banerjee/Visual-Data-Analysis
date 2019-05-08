#!/usr/bin/env python
# coding: utf-8

# In[119]:


# Name- 1. Barshana Banerjee, 2. Hemanth Kumar Reddy Mayaluru

from PIL import Image,ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt   

#Display Function
def show_image(title, image):
    cv2.imshow(title, image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def rgb_hsv(imge):
    hsv=cv2.cvtColor(imge, cv2.COLOR_BGR2HSV)
    return hsv

def hsv_rgb(imge):
    rgb=cv2.cvtColor(imge, cv2.COLOR_BGR2HSV)
    return rgb.astype('uint8')

def shift_hue(arr,hout):
    hsv=rgb_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_rgb(hsv)
    return rgb


# Task(a) reads image as RGB 
img = cv2.imread('oldtimer.png') 
# shows the image 
show_image("Original Image", img)

#Task(b) Convert image to HSV by using cv2color 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
# Shows the image 
show_image('HSV Image', img_hsv)  

#Convert image to GRAY by retaining the Value
h, s, v = cv2.split(img_hsv)
show_image("gray-image",v)


#Task(c) Convert image to GRAY by using cv2color 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
# Shows the image 
show_image('Gray Image CV2', img_gray)  

#Task(d) - Saturation
img_orig = Image.open('oldtimer.png')
converter = ImageEnhance.Color(img_orig)
img_sat = converter.enhance(0.5)
img_sat.show()

#Task(e) - Image Blend
img_brown = cv2.applyColorMap(numpy.array(img_sat), cv2.COLORMAP_PINK)
show_image("Image Brown Blend", img_brown)

#Task(f)- its not working
'''orig_img = Image.open('oldtimer.png').convert('RGBA')
arr = np.array(orig_img)
green_hue = (180-78)/360.0
blue_hue = 240
new_img = Image.fromarray(shift_hue(arr,blue_hue), 'RGBA')
new_img.show()
new_img = Image.fromarray(shift_hue(arr,green_hue), 'RGBA')
new_img.show()

'''


# In[103]:





# In[ ]:





# In[102]:





# In[ ]:





# In[ ]:




