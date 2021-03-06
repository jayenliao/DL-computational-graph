# Deep Learning - HW2: Computational Graph

Author: Jay Liao (re6094028@gs.ncku.edu.tw)

This is assignment 2 of Deep Learning, a course at Institute of Data Science, National Cheng Kung University. This project aims to extract features from images and to construct models to perform image classification.

## Data

### The raw data
    
- Images: please go to https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr to download raw image files and put them under the folder `./images/`. There are 64,225 files with 50 subfolders.

- File name lists of images: `./data/train.txt`, `./data/val.txt`, and `./data/test.txt`.

### The processed data (extracted image features)

- There were 3 methods of feature extraction utilized here. Since it cost much time to generate feature matrices, they have been produced in advance and placed in the folder `./data/`.

    1. Histogram:  `./data/X_Histogram_tr.npy`,  `./data/X_Histogram_va.npy`, and  `./data/X_Histogram_te.npy`. We use the concept of the global color histogram. Given the number of ranges (bin cuts) of the histogram, count the no. of each range for 3 channels respectively. Thus 15 features would be obtained if set the no. of ranges as 5, for example. Since the input images are not in the same shape, we took the mean values.
 
    2. Scale-invariant feature transform (SIFT): `./data/X_SIFT_tr.npy` (should be produced by running the code),  `./data/X_SIFT_va.npy`, and  `./data/X_SIFT_te.npy`. Package `OpenCV` was utilized to take key points and descriptors with SIFT method. Since the input images are not in the same shape,  we took column means as features for each image.

    3. Speeded Up Robust Features (SURF):  `./data/X_SURF_tr.npy` (should be produced by running the code),  `./data/X_SURF_va.npy`, and  `./data/X_SURF_te.npy`. Package `OpenCV` was utilized to take key points and descriptors with SURF method. Since the input images are not in the same shape,  we took column means as features for each image.

- If you want to see the process of feature extraction, just easily remove these feature matrices files from the folder before running. 

## Code

- Source codes:

    - `./source/utils.py`: little tools

    - `./source/feature_extraction.py`: functions for feature extraction

    - `./source/layers.py`: layers for NN model construction, e.g., `ReLU()`, `Sigmoid()`

    - `./source/models.py`: model construction
    
    - `./source/trainers.py`: class for training, predicting, and evaluating the models

- `args.py`: define the arguments parser

- `main.py`: the main program with loading, training, and evaluating procedures.

- `exp_hidden_act.py`: experiment program to compare different hidden activation functions.

-  `experiments.ipynb`: experiments results

- `requirements.txt`: required packages

## Folders

- `./images/` should contain raw image files (please go to download and put them with subfolders here).

- `./data/` contains .txt files of image lists and .npy files of extracted feature matrices. Some extracted feature matrices of training set are not uploaded here becuase they too large. Please run the code to produce them.

- `./output/` will contain trained models, model performances, and experiments results after running. 

## Requirements

```
numpy==1.16.3
pandas==1.1.5
opencv_python==3.4.2.16
tqdm==4.50.0
matplotlib
```

## Usage

1. Clone this repo.

```
git clone https://github.com/jayenliao/DL-computational-graph.git
```

2. Set up the required packages.

```
cd DL-computational-graph
pip3 install requirements.txt
```

3. Run the experiments.

```
python3 main.py
python3 exp_hidden_act.py
```

It may take much time to run the whole `main.py`. The arguments parser can be used to run several experiments only, such as:

```
python3 main.py --models 'TwoLayerPerceptron' --epochs 500 --savePATH './output/_TwoLayerPerceptron'
python3 exp_hidden_act.py --verbose --savePATH './output/exp_hidden_act' --feature_types 'SIFT' --epochs 500 --print_result_per_epochs 100
```

You can also directly access the experiments results on `experiments.ipynb`.

## Reference

1. Lowe, D. G. (1999, September). Object recognition from local scale-invariant features. In Proceedings of the seventh IEEE international conference on computer vision (Vol. 2, pp. 1150-1157). Ieee.

2. Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008). Speeded-up robust features (SURF). Computer vision and image understanding, 110(3), 346-359.

3. ?????????????????????????????????2017??????Deep Learning: ???Python??????????????????????????????????????????*??????????????????????????????*???ISBN: 9789864764846???GitHub: https://github.com/oreilly-japan/deep-learning-from-scratch ???

4. Watt, J., Borhani, R., & Katsaggelos, A. K. (2019). Machine learning refined. ISBN: 9781107123526. GitHub: https://github.com/jermwatt/machine_learning_refined.
