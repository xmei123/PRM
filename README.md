# PRM
Reproduce Zhou's work on DGX-a using multiple GPUs

1. Create a docker image using pytorch 18.03. Pull /home/mei/datasets to the creased docker image

2. Configure Zhou's enviroment in docker. Follow Zhou's github instructions

3. Change batch_size = 9 in /PRM-pytorch/demo/config.yml to have batch size greater than the number of GPUs as well as not to consume too much memory. (may change to the default batch_size = 16 later, I used five GPUs when I ran the code).




一.在DGX-1上创建新的docker环境，pytorch版本为18.03,将下载在/home/mei目录下的datasets映像到docker中。

二.在docker中配置zhou环境，具体按照zhou的github上的说明

三.将/PRM-pytorch/demo/config.yml中的batch_size改为9，防止太大内存不够以及使batch_size大于GPU数量

四.测试data中的sample文件，正确
