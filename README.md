# CCNet Document
[中文文档](README_zh.md)

## About Model

The model backbone is CSP-Darknet, the feature extraction network is PANet, and the detection head network is Decoupled head.

**We provide the following network structures**

1. `net.py` include SPPF, SpaceToDepth, MobileViT
2. `net_old.py` include SPPF, MobileViT
3. `net_deploy.py` include SPPF, FocusReplaceConv

**What improvements have we made to the overall network?**

1. We replaced the SPP of layer5 of CSP-Darknet with SPPF, and added MobileViT after SPPF (Our implementation based on Apple's paper: [《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》](https://arxiv.org/abs/2110.02178))
2. In the backbone network, we set a (1: 1: 3: 1) ratio for Bottleneck in the last 4 CSPLayer, references: [《A ConvNet for the 2020s》](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)
3. We added a three-layer CBAM attention mechanism between the backbone network and the feature extraction network
4. We adjusted the downsample structure of the overall network and used a new downsampling method to reduce the loss of granularity, references: [No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects](https://arxiv.org/abs/2208.03641)



## Coming Soon

- [ ] Enhance the recognition of small target objects
- [ ] Add COCO format training files
- [ ] Development of the TensorRT version



## How to Train and Evaluate

1. Put the dataset into datasets and create a new folder for it (all paths do not use Chinese), you need to use VOC format (JPEGImages, Annotations)
2. About generating datasets, modify `voc_classes`, `file_dir` in `tools/annotations.py`, right-click to run
3. Regarding training, modify the parameters in `if __name__ == "__main__"` in `tools/train.py`, right-click to run the training
4. For training datasets, enter `tensorboard --logdir=results` in the console, open the browser (**do not use Google, there will be bugs**), enter `localhost:6006` to view;
5. About evaluation, modify the parameters in `if __name__ == "__main__"` in `tools/eval.py`, right-click to run the evaluation



## How to Communicate with Servers such as Springboot

1. In the `tools/socket2springboot.py` file, we use sockets for communication. The default port of the project is `12345`. Before deploying to the server, please install the corresponding working environment of the project (Anaconda, Miniconda, etc.). Set the working environment to be enabled by default. For example, take Miniconda as an example, use the `sudo vim .bashrc` command to open the `.bashrc` file, add `conda activate XXX (your Pytorch environment)` in the last line, save it and use `source .bashrc` command to complete the configuration

2. The project provides a script file, which runs as `bash socket2springboot.sh`

3. Use the `nohup python tools/socket2springboot.py` command in the current working environment to run. We can save the python output log using the `nohup python -u tools/socket2springboot.py > /your/path/log/socket2springboot.log 2>&1 &` command. If you need to restart or shut down the service, use the `htop` command to find the running program and kill it

4. When the test runs successfully, we use the `ps -def | grep "socket2springboot.py"` command to view the pid of the current program. When it can be found, run `python test/method_test.py` to send the test Json data stream
   
5. The data stream uses the Json format, and the format from the client to the server is as follows:

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
6. When the data is sent, we use the `vim log/socket2springboot.log` command to open the log file, and the Json data stream returned by the server will be recorded in the file

7. The data stream is returned from the server to the client in the following format:

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