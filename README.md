# Image Similarity Finder App
DSAN6600 Final Project

[![deeplake](https://img.shields.io/badge/powered%20by-Deep%20Lake%20-ff5a1f.svg)](https://github.com/activeloopai/deeplake)


## Team 1 Members:
- [Austin Barish](https://github.com/austinbarish)
- [Landon Carpenter](https://github.com/lecarpen23)
- [Minsuh (Eric) Lim](https://github.com/5cminsuhlim)
- [Nolan Penoyer](https://github.com/NolanPenoyer)
- [Shawn Xu](https://github.com/shawnhxu)

## Introduction

Inspired by [@ArtButSports](https://twitter.com/ArtButSports?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) on Twitter, our team engineered an app that will take in a users' input in the form of an image and output the most similar art image or images. We hope our app enables anyone to find the most similar art piece(s) to their input image!

## Dataset
- [WikiArt through DeepLake open-source](https://datasets.activeloop.ai/docs/ml/datasets/wiki-art-dataset/)

## How to Download Vector Stores:
Since our pool of art images and their embeddings are too large in size, we have a Google Drive link to it [here](https://drive.google.com/drive/folders/1A8W3FmveqRz5ne98V-A62LooueEQQNm4?usp=sharing).

Due to its size, downloading from Google Drive results in many separate zip files. To extract them all do the following:
1) Create a folder directory and store the separate zips.
2) Create an output folder directory.
3) In terminal of choice, run ```unzip '*.zip' -d 'output_directory_path'``` while being in stored folder directory.
4) This will result in a single vector_stores folder containing all vector stores with each preprocessing method.

## How to use App:
1) Put new vector_stores folder in the data folder
2) Set current directory to the main directory.
3) Run app with `streamlit run app/streamlit_app.py`
4) In local browser, select an input image and adjust parameters to your liking.
5) Then click "Find Similar Images".
6) To create a consolidated output, select desired preprocessing type and click "Create Output Image"

## References

- He, Kaiming, et al. “Deep residual learning for image recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016, https://doi.org/10.1109/cvpr.2016.90.
- Redmon, Joseph, et al. “You only look once: Unified, real-time object detection.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016, https://doi.org/10.1109/cvpr.2016.91. 
- Michael Danielczuk, Matthew Matl, Saurabh Gupta, Andrew Li, Andrew Lee, Jeffrey Mahler, and Ken Goldberg. "Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data." In *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2019.
- WikiArt Emotions: An Annotated Dataset of Emotions Evoked by Art. Saif M. Mohammad and Svetlana Kiritchenko. In Proceedings of the 11th Edition of the Language Resources and Evaluation Conference (LREC-2018), May 2018, Miyazaki, Japan.
