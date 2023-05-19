## 关于模型

本模型主干网络为CSP-Darknet，特征提取网络为PANet，检测头网络为Decoupled head。

**我们对整体网络有哪些改进？**

1. 我们将CSP-Darknet的layer5的SPP换成了SPPF，在SPPF后加入了MobileViT（我们基于苹果的论文所复现的[《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》](https://arxiv.org/abs/2110.02178)）；
2. 在主干网络中，我们对后4个CSPLayer中的Bottleneck设置了（1: 1: 3: 1）的比例，参考论文：[《A ConvNet for the 2020s》](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)；
3. 在主干网络与特征提取网络的中间我们加入了三层CBAM注意力机制；
4. 我们将整体网络的下采样结构进行了调整，使用全新的下采样方法，减少颗粒度损失。



## 未来的工作

- [x] 加强对小目标物体的识别
- [ ] 增加COCO格式训练文件
- [ ] TensorRT版本的开发
- [x] 增加AMFF-YOLOX结构


## 如何训练与评估

1. 将数据集放到datasets中，并为其创建一个新的文件夹（所有路径请勿使用中文），需要使用VOC格式（JPEGImages、Annotations）；
2. 关于生成数据集，在`tools/annotations.py`修改`voc_classes`、`file_dir`，右键运行；
3. 关于训练，在`tools/train.py`修改`if __name__ == "__main__"`中的参数，右键运行训练即可；
4. 训练数据在控制台中输入`tensorboard --logdir=results`即可，打开浏览器，输入`localhost:6006`查看；
5. 关于评估，在`tools/eval.py`修改`if __name__ == "__main__"`中的参数，右键运行评估即可。

## 前置部署

1. 我们默认使用者使用Linux，安装Anaconda、Miniconda等虚拟环境，所以这里不做过多赘述。如有不会的请自行询问搜索引擎或ChatGPT

2. 为你的虚拟环境创建一个新的环境或使用base环境，在项目目录中找到`auto_install_package.sh`文件，使用命令进行本项目所需包的安装

   ```bash
   sh auto_install_package.sh
   ```

3. 将工作环境设置为默认开启，例如以Miniconda为例，使用命令打开`.bashrc`文件

   ```bash
   sudo vim ~/.bashrc
   ```

   在最后一行加入以下语句

   ```bash
   conda activate XXX(你的Pytorch环境)
   ```

   保存后使用命令即可完成配置

   ```bash
   source ~/.bashrc
   ```

## 如何进行与Springboot等服务器进行交互

1. 在`tools/socket2springboot.py`文件中，我们使用socket进行通信。项目默认端口为`12345`

2. 项目提供自动部署脚本文件`socket2springboot.sh`，使用前请先修改脚本文件中的`project_path`，运行方式如下

   ```bash
   sh socket2springboot.sh
   ```

3. 当然你也可以手动部署，在当前工作环境中使用以下命令即可运行，此命令仅后台运行

   ```bash
   nohup python tools/socket2springboot.py
   ```

   我们也可以使用命令将python后台运行并将输出日志保存到指定文件

   ```bash
   nohup python -u tools/socket2springboot.py > /your/path/log/logname.log 2>&1 &
   ```

   如果需要重启或关闭服务，使用`htop`命令找到运行程序后Kill，或使用指令kill

   ```bash
   kill $(ps -ef | grep "socket2springboot.py" | grep -v grep | awk "{print $2}")
   ```

4. 当测试是否运行成功时，我们使用命令查看当前程序运行的pid

   ```bash
   ps -def | grep "socket2springboot.py"
   ```

   当其可以被找到后，运行`method_test.py`发送测试的Json数据流

   ```bash
   python test/method_test.py
   ```

   

5. 数据流传输均使用Json格式，由客户端传到服务端格式如下

   ```json
   {
       "model_name": "Net", 
       "model_path": "/your/modle/path/your_model_name.pth",
       "dataset": "NEUDET",
       "input_shape": [224, 224],
       "conf_thres": 0.5,
       "nms_thres": 0.6,
       "image_path": ["/your/image/path/image1.jpg",
                      "/your/image/path/image2.jpg",
                      "/your/image/path/image3.jpg"]
   }
   ```
6. 当发送完数据后，我们使用命令打开log文件，此时文件中将记录服务端返回Json数据流

   ```bash
   vim log/socket2springboot_timestamp.log
   ```

7. 数据流由服务端返回到客户端格式如下

   ```json
   {
       "image": [
           {"image_path": "/your/image/path/image1.jpg", "image_id": 0}, 
           {"image_path": "/your/image/path/image2.jpg", "image_id": 1}, 
           {"image_path": "/your/image/path/image3.jpg", "image_id": 2}], 
       "annotations": [
           {"image_id": 0, "box": [117.38097, 32.385307, 133.57707, 62.68074], "predicted_class": "class1", "conf": 0.66143817}, 
           {"image_id": 0, "box": [33.51411, 6.4222217, 48.198418, 34.87923], "predicted_class": "class1", "conf": 0.64585626}, 
           {"image_id": 0, "box": [112.736916, 146.3184, 133.77809, 198.33405], "predicted_class": "class1", "conf": 0.63799584}, 
           {"image_id": 0, "box": [114.86066, 73.44805, 142.24493, 114.508995], "predicted_class": "class1", "conf": 0.61806077}, 
           {"image_id": 1, "box": [108.89356, -5.1439524, 180.93867, 138.25955], "predicted_class": "class2", "conf": 0.7961505}, 
           {"image_id": 1, "box": [92.11259, 127.67179, 183.05533, 199.58357], "predicted_class": "class2", "conf": 0.7741741}, 
           {"image_id": 1, "box": [10.045683, 65.02976, 73.03552, 122.30415], "predicted_class": "class2", "conf": 0.72269356}, 
           {"image_id": 2, "box": [32.543327, 38.372814, 145.7598, 195.50223], "predicted_class": "class3", "conf": 0.7584684}]
   }
   
   ```

## 项目具体实施界面展示

该界面编写作者为@[lizhaoguo123](https://github.com/lizhaoguo123)

**检测界面**

![image-20221209010633957](assets/image-20221209010633957.png)

![image-20221209011503907](assets/image-20221209011503907.png)

**管理员界面**

![image-20221209010009245](assets/image-20221209010009245.png)

![image-20221209010021494](assets/image-20221209010021494.png)

![image-20221209010031676](assets/image-20221209010031676.png)

![image-20221209010224632](assets/image-20221209010224632.png)

![image-20221209010147174](assets/image-20221209010147174.png)

![image-20221209010607772](assets/image-20221209010607772.png)