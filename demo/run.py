
# coding: utf-8

# ## Weakly Supervised Instance Segmentation using Class Peak Response 
# ### Demo code

# In[1]:





# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import os
import json
#import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imresize


# Access PRM through [Nest](https://github.com/ZhouYanzhao/Nest)

# In[4]:


from nest import modules, run_tasks


# ### Train a PRM-augmented classification network using image-level labels

# In[5]:

def main():
	class_names = modules.pascal_voc_object_categories()
	print('Object categories: ' + ', '.join(class_names))
	image_size = 448
	#image pre-processor
	transformer = modules.image_transform(
	image_size = [image_size, image_size],
	augmentation = dict(),
	mean = [0.485, 0.456, 0.406],
	std = [0.229, 0.224, 0.225])
	backbone = modules.fc_resnet50(num_classes=14, pretrained=False)
	print(backbone)
	model = modules.peak_response_mapping(backbone)
	model = nn.DataParallel(model)
	state = torch.load('./snapshots/model_latest.pt')
	# model.load_state_dict(state['model'])
	model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./snapshots/model_latest.pt')['model'].items()})
	model = model.module.cuda()
	print(transformer)
if __name__ =='__main__':
	main()
