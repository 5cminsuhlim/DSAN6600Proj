# Image Similarity Finder App
DSAN6600 Final Project

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
3) In terminal of choice, run `unzip '*.zip' -d 'output_directory_path'` while being in stored folder directory.
4) This will result in a single vector_stores folder containing all vector stores with each preprocessing method.

## How to use App:
1) Set current directory to the main directory.
2) Run app with `streamlit run app/streamlit_app.py`
3) In local browser, select an input image and adjust parameters to your liking.
4) Then click "Find Similar Images".
5) To create a consolidated output, select desired preprocessing type and click "Create Output Image"



[![deeplake](https://img.shields.io/badge/powered%20by-Deep%20Lake%20-ff5a1f.svg)](https://github.com/activeloopai/deeplake)
