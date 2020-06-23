# ECML-PKDD 2020

This repository is the official implementation of MMCNN: A Multi-branch Multi-scale Convolutional Neural Network for Motor Imagery Classification.
 * The Model architecture
  ![Alt text](/image/MMCNN.png)  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
```
* keras
* tensorflow
* numpy
* scipy
* pylab
* sklearn
* random
```

## Dataset and Data Preparation
We Evaluated our model using [BCI Competition IV 2a and 2b dataset](http://www.bbci.de/competition/iv/#dataset2a) and we have cut the main part of the data which have been transformed to the .npy file.

## Files Information
#### Dataprocess.py
This file is used to load datasets and augment data.
#### MMCNN_model.py
This file is used to implement the functions in the model.
#### Evaluation.py
This file is used to evaluate the model.
#### main.py
The main file to run the model.
## Training And Evaluation

We conducted experiments under

- python 3.7.3
- tensorflow 1.13.1
- kaggle gpu

To train the model(s) in the paper, run this command:

```
train:
eg1:python main.py --get2apath bci2a-npy/ --choosedata 2a
eg2:python main.py --get2bpath bci2b-npy/ 
```

The commands above are examples to train the model with dataset 2a or 2b.The evaluation is follow the training process.The final results are 5-fold cross-validation with all datas from all subjects.If you want to train model with 
a centain one subject,you can change in the main.py file.


## Results

Our model achieves the following performance :
* The Performance on dataset 2a
  ![Alt text](/image/2a.png)  
* The Performance on dataset 2b
  ![Alt text](/image/2b.png)  
