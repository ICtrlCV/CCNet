## 关于模型

本模型主干网络为CSP-Darknet，特征提取网络为PANet，检测头网络为Decoupled head。

**我们对整体网络有哪些改进？**

1. 我们将CSP-Darknet的layer5的SPP换成了SPPF，在SPPF后加入了MobileViT（我们基于苹果的论文所复现的[《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》](https://arxiv.org/abs/2110.02178)）；
2. 在主干网络中，我们对后4个CSPLayer中的Bottleneck设置了（1: 1: 3: 1）的比例，参考论文：[《A ConvNet for the 2020s》](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)；
3. 在主干网络与特征提取网络的中间我们加入了三层CBAM注意力机制；
4. 我们将整体网络的下采样结构进行了调整，使用全新的下采样方法，减少颗粒度损失。



## 未来的工作

- [ ] 加强对小目标物体的识别
- [ ] 增加COCO格式训练文件
- [ ] TensorRT版本的开发



## 如何训练与评估

1. 将数据集放到datasets中，并为其创建一个新的文件夹（所有路径请勿使用中文），需要使用VOC格式（JPEGImages、Annotations）；
2. 关于生成数据集，在tools/annotations.py修改voc_classes、file_dir，右键运行；
3. 关于训练，在tools/train.py修改`if __name__ == "__main__"`中的参数，右键运行训练即可；
4. 训练数据在控制台中输入`tensorboard --logdir=results`即可，打开浏览器**（不要用谷歌，会出bug）**，输入`localhost:6006`查看；
5. 关于评估，在tools/eval.py修改`if __name__ == "__main__"`中的参数，右键运行评估即可。



## 如何进行与Springboot等服务器进行交互

1. 在utils/socket2springboot.py文件中，我们使用socket进行通信。项目默认端口为`12345`，在部署到服务器之前，请先安装项目相应的工作环境（Anaconda、Miniconda等）。将工作环境设置为默认开启，例如以Miniconda为例，使用`sudo vim .bashrc`命令打开`.bashrc`文件，在最后一行加入`conda activate XXX(你的Pytorch环境)`，保存后使用`source .bashrc`命令即可完成配置；

2. 在当前工作环境中使用`nohub python utils/socket2springboot.py`命令即可运行。我们可以使用`nohub python utils/socket2springboot.py > ../log/socket2springboot.log`中。如果需要重启或关闭服务，使用`htop`命令找到运行程序后Kill；

3. 数据流传输均使用Json格式，由客户端传到服务端格式如下：

   ```json
   {
       "model_name": "Net", 
       "model_path": "../results/1662448384.497914/model_200.pth",
       "dataset": "NEUDET",
       "input_shape": [224, 224],
       "conf_thres": 0.5,
       "nms_thres": 0.6,
       "image_path": ["../asserts/inclusion_1.jpg",
                      "../asserts/patches_235.jpg",
                      "../asserts/rolled-in_scale_264.jpg"]
   }
   ```

4. 数据流由服务端返回到客户端格式如下：

   ```json
   {
       "image": [
           {"image_path": "../asserts/inclusion_1.jpg", "image_id": 0}, 
           {"image_path": "../asserts/patches_235.jpg", "image_id": 1}, 
           {"image_path": "../asserts/rolled-in_scale_264.jpg", "image_id": 2}], 
       "annotations": [
           {"image_id": 0, "box": [117.38097, 32.385307, 133.57707, 62.68074], "predicted_class": "inclusion", "conf": 0.66143817}, 
           {"image_id": 0, "box": [33.51411, 6.4222217, 48.198418, 34.87923], "predicted_class": "inclusion", "conf": 0.64585626}, 
           {"image_id": 0, "box": [112.736916, 146.3184, 133.77809, 198.33405], "predicted_class": "inclusion", "conf": 0.63799584}, 
           {"image_id": 0, "box": [114.86066, 73.44805, 142.24493, 114.508995], "predicted_class": "inclusion", "conf": 0.61806077}, 
           {"image_id": 1, "box": [108.89356, -5.1439524, 180.93867, 138.25955], "predicted_class": "patches", "conf": 0.7961505}, 
           {"image_id": 1, "box": [92.11259, 127.67179, 183.05533, 199.58357], "predicted_class": "patches", "conf": 0.7741741}, 
           {"image_id": 1, "box": [10.045683, 65.02976, 73.03552, 122.30415], "predicted_class": "patches", "conf": 0.72269356}, 
           {"image_id": 2, "box": [32.543327, 38.372814, 145.7598, 195.50223], "predicted_class": "rolled-in_scale", "conf": 0.7584684}]
   }
   
   ```