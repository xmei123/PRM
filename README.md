# PRM
Reproduce Zhou's work on DGX-a using multiple GPUs

1. Create a docker image using pytorch 18.03. Pull /home/mei/datasets to the creased docker image

2. Configure Zhou's enviroment in docker. Follow Zhou's github instructions

3. Change batch_size = 9 in /PRM-pytorch/demo/config.yml to have batch size greater than the number of GPUs as well as not to consume too much memory. (may change to the default batch_size = 16 later, I used five GPUs when I ran the code).

4. Under /Nest-pytorch/install/trainer.py, line 87, add device_ids 0 to 7
#device_ids = [0,1,2,3,4,5,6,7]

5. Under /PRM-pytorch/demo/main.py, add two lines plt.savefig('./test.png') and plt.savefig('./test2.png') to save the tested sample data (had trouble displaying images in server).

6. I copied the two sample results test.png and test2.png to my dirctory /home/xmei, which are also showed below




