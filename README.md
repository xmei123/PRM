# PRM
Reproduce Zhou's work on DGX-a using multiple GPUs

1. Create a docker image named PRM using pytorch 18.03. Pull /home/mei/datasets to the created docker image

2. Configure Zhou's enviroment in docker. Follow Zhou's github instructions to install nest etc.

3. Change batch_size = 9 in /PRM-pytorch/demo/config.yml to have batch size greater than the number of GPUs as well as not to consume too much memory. (may change to the default batch_size = 16 later, I used five GPUs when I ran the code).

4. Under /Nest-pytorch/install/trainer.py, line 87, add device_ids 0 to 7
#device_ids = [0,1,2,3,4,5,6,7]

5. Under /PRM-pytorch/demo/main.py, add two lines plt.savefig('./sample_peak_map.png') and plt.savefig('./sample_prediction.png') to save the tested sample data (had trouble displaying images in server).

6. I copied all sample results sample_peak_map.png and sample_prediction.png to my dirctory /home/xmei in server, which are also showed below


steps:
(1) docker start PRM
(2) docker attach PRM
(3) cd PRM-pytorch/demo
(5) python main.py
 ![加载图片](https://github.com/xmei123/PRM/blob/master/sheep.png)
 ![加载图片](https://github.com/xmei123/PRM/blob/master/sheep%20(2).png)




