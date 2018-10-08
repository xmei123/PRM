import warnings
warnings.filterwarnings("ignore")


# In[3]:


import os
import json
import datasets
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


def main():
	class_names = modules.pascal_voc_object_categories()
	image_size = 448
	# image pre-processor
	transformer = modules.image_transform(
		image_size = [image_size, image_size],
		augmentation = dict(),
		mean = [0.485, 0.456, 0.406],
		std = [0.229, 0.224, 0.225])
	backbone = modules.fc_resnet50(num_classes=20, pretrained=False)
	model = modules.peak_response_mapping(backbone)
	# loaded pre-trained weights
	model = nn.DataParallel(model)
	state = torch.load('./snapshots/model_latest.pt')
	model.load_state_dict(state['model'])
	model = model.module.cuda()
	idx = 1
	raw_img = PIL.Image.open('./data/sample%d.jpg' % idx).convert('RGB')
	input_var = transformer(raw_img).unsqueeze(0).cuda().requires_grad_()
	with open('./data/sample%d.json' % idx, 'r') as f:
		proposals = list(map(modules.rle_decode, json.load(f)))

	# plt.savefig('./%d result.jpg' %idx)
	model = model.eval()
	confidence = model(input_var)
	print('Object categories in the image:')
	confidence = model(input_var)
	for idx in range(len(class_names)):
		if confidence.data[0, idx] >0:
			print('    [class_idx: %d] %s (%.2f)' % (idx, class_names[idx], confidence[0, idx]))

	model = model.inference()

	visual_cues = model(input_var)
	if visual_cues is None:
		print('No class peak response detected')
	else:
		confidence, class_response_maps, class_peak_responses, peak_response_maps = visual_cues
		_, class_idx = torch.max(confidence, dim=1)
		class_idx = class_idx.item()
		num_plots = 2 + len(peak_response_maps)
		f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))    #表示一行四列
		axarr[0].imshow(imresize(raw_img, (image_size, image_size), interp='bicubic'))
		axarr[0].set_title('Image')
		axarr[0].axis('off')
		axarr[1].imshow(class_response_maps[0, class_idx].cpu(), interpolation='bicubic')
		axarr[1].set_title('Class Response Map ("%s")' % class_names[class_idx])
		axarr[1].axis('off')
		for idx, (prm, peak) in enumerate(sorted(zip(peak_response_maps, class_peak_responses), key=lambda v: v[-1][-1])):
			axarr[idx + 2].imshow(prm.cpu(), cmap=plt.cm.jet)
			axarr[idx + 2].set_title('Peak Response Map ("%s")' % (class_names[peak[1].item()]))
			axarr[idx + 2].axis('off')
		plt.savefig('./test.png')
	instance_list = model(input_var, retrieval_cfg=dict(proposals=proposals, param=(0.95, 1e-5, 0.8)))

	# visualization
	if instance_list is None:
		print('No object detected')
	else:
		# peak response maps are merged if they select similar proposals
		vis = modules.prm_visualize(instance_list, class_names=class_names)
		f, axarr = plt.subplots(1, 3, figsize=(12, 5))
		axarr[0].imshow(imresize(raw_img, (image_size, image_size), interp='bicubic'))
		axarr[0].set_title('Image')
		axarr[0].axis('off')
		axarr[1].imshow(vis[0])
		axarr[1].set_title('Prediction')
		axarr[1].axis('off')
		axarr[2].imshow(vis[1])
		axarr[2].set_title('Peak Response Maps')
		axarr[2].axis('off')
		plt.savefig('./test2.png')
		plt.show()
if __name__ =='__main__':
    main()