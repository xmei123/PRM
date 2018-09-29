# PRM
重现zhou的代码并且使用DGX-1多GPU训练。

一.在DGX-1上创建新的docker环境，pytorch版本为18.03,将下载在/home/mei目录下的datasets映像到docker中。

二.在docker中配置zhou环境，具体按照zhou的github上的说明

三.将/PRM-pytorch/demo/config.yml中的batch_size改为9，防止太大内存不够以及使batch_size大于GPU数量

四.测试data中的sample文件，正确
