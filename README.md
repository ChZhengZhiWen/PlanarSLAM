# Planar_SLAM_Semidirect
This repo proposes a RGB-D SLAM system specifically designed for structured environments and aimed at improved tracking and mapping accuracy by relying on geometric features that are extracted from the surrounding. More details can be found in our papers ([RGB-D](https://arxiv.org/abs/2010.07997) and [Monocular](https://arxiv.org/abs/2008.01963)).  

**Authors:** Zhiwen Zheng, He Wang, Ru Li

### updated on 11.12.2021

We updated the planar meshing section by making use of a more efficient triangulation method, rather than the greedy algorithm from PCL. If the PCL window shows nothing, maybe you could click "R" after selecting the window. 

ps: the reconstruction method is still very naive, we will keep moving.



----
## License

PlanarSLAM is released under a GPLv3 license.

For commercial purposes, please contact the authors: yanyan.li (at) tum.de. If you use PlanarSLAM in an academic work, please cite:

```
inproceedings{Li2021PlanarSLAM,
  author = {Li, Yanyan and Yunus, Raza and Brasch, Nikolas and Navab, Nassir and Tombari, Federico},
  title = {RGB-D SLAM with Structural Regularities},
  year = {2021},
  booktitle = {2021 IEEE international conference on Robotics and automation (ICRA)},
 }
```

## 1. Prerequisites

We have tested the library in **ubuntu 16.04** and **ubuntu 18.04**, but it should be easy to compile on other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

### Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### OpenCV and **OpenCV_Contrib**
We use [OpenCV](http://opencv.org) and corresponding **OpenCV_Contrib** to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Tested with OpenCV 3.4.1**

### Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

### DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

### PCL

We use [PCL](http://www.pointclouds.org/) to reconstruct and visualize mesh. Download and install instructions can be found at: https://github.com/ros-perception/perception_pcl. **Tested with PCL 1.7.0 and 1.9.0**.

1. https://github.com/PointCloudLibrary/pcl/releases

2. compile and install

   ```
   cd pcl 
   mkdir release 
   cd release
   
   cmake  -DCMAKE_INSTALL_PREFIX=/usr/local \ -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON \ -DCMAKE_INSTALL_PREFIX=/usr/local -DPCL_DIR=/usr/local/share/pcl  .. 
   
   make -j12
   sudo make install
   ```



## 2. Test the system

### Structural Public Scenes

[ICL NUIM](http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), [Structural TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset/download), All types of Corridors

### Test the system locally

1. Download **'freiburg3_structure_texture_far'** and  associate RGB-D pairs based on [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools) provided by the dataset.

   ```
   python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
   ```

2. Compile the system

```
./build.sh
```

​	3.  Run the system

```
./Examples/RGB-D/Planar_SLAM Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml PATH_TO_SEQUENCE_FOLDER .PATH_TO_SEQUENCE_FOLDER/ASSOCIATIONS_FILE
```

*similar command for testing ICL-NUIM sequences*

```
./Examples/RGB-D/Planar_SLAM Vocabulary/ORBvoc.txt Examples/RGB-D/ICL.yaml PATH_TO_SEQUENCE_FOLDER  PATH_TO_SEQUENCE_FOLDER/ASSOCIATIONS_FILE

```



----

## Citation
```
inproceedings{Li2021PlanarSLAM,
  author = {Li, Yanyan and Yunus, Raza and Brasch, Nikolas and Navab, Nassir and Tombari, Federico},
  title = {RGB-D SLAM with Structural Regularities},
  year = {2021},
  booktitle = {2021 IEEE international conference on Robotics and automation (ICRA)},
 }
```

## Acknowledgement

ORB_SLAM2 and the corresponding community.

slam基础环境配置

1.安装Ubuntu18.04双系统   具体可参考 https://blog.csdn.net/qq_42257666/article/details/123709678

2.安装g++、git（gcc系统自带） sudo apt-get install g++ git

3.源码安装cmake  cmake官网（https://www.baidu.com/link?url=h4ytas70nqIij9PreyTc6d3lP9tRqdGQnGYGI9gMHF7&wd=&eqid=e38ed27c00027a190000000664167523）
首先安装依赖 sudo apt-get install build-essential libssl-dev
解压后进入cmake  ./bootstrap
make -j12
sudo make install

4.安装eigen3  sudo apt-get install libeigen3-dev  通过命令行安装的eigen版本为 3.3.4

5.源码安装Pangolin   依赖安装 sudo apt-get install libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev
下载源码 git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cmake .. && make -j12 && sudo make install
测试是否安装成功  cd Pangolin/build/examples/HelloPangolin && ./HelloPangolin
出现彩色正方体则安装成功

6.安装opecv3.4.1与opencv_contrib3.4.1  详细过程参考  https://blog.csdn.net/xhtchina/article/details/126422425?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167919381716800184198516%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=167919381716800184198516&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~times_rank-2-126422425-null-null.142^v74^insert_down4,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=ubuntu18%E5%AE%89%E8%A3%85opencv3.4.1%E4%B8%8Eopencv%20contrib&spm=1018.2226.3001.4187
安装过程遇到的问题请参考  https://blog.csdn.net/qq_57061492/article/details/127873444?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167919381716800184198516%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=167919381716800184198516&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~times_rank-1-127873444-null-null.142^v74^insert_down4,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=ubuntu18%E5%AE%89%E8%A3%85opencv3.4.1%E4%B8%8Eopencv%20contrib&spm=1018.2226.3001.4187
测试是否安装成功   cd ../samples/cpp/example_cmake
cmake .
make
./opencv_example
电脑摄像头打开即为安装成功

7.安装pcl1.9.0   具体可参考 https://blog.csdn.net/qq_42257666/article/details/124574029?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167919417216800180641165%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=167919417216800180641165&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~times_rank-8-124574029-null-null.142^v74^insert_down4,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=ubuntu18%E5%AE%89%E8%A3%85pcl1.9.1&spm=1018.2226.3001.4187
建议pcl1.9.0 + vtk7.1.1    vtk编译出现找不到qt的错误时   sudo apt-get install qt4-default