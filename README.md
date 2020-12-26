# AEPI: Automated Ear Pinna Identification.

![teaser](https://github.com/UsamaHasan/AEPI-Automated-Ear-Pinna-Identification/blob/master/doc/spnet.png)

Code release for the paper AEPI: Representation Learning and Evaluation of Human Ear Identification based on a blend of Residual Network and Spatial Encoding.

**Authors**:[Usama Hasan](https://usamahasan.github.io/) ,[Waqar Hussain](https://www.researchgate.net/profile/Waqar_Hussain7),[Nouman Rasool](https://www.researchgate.net/profile/Nouman_Rasool)

## Introduction
In this work, we present automated ear identification model on Ear VN dataset, a large-scale ear images dataset in the wild.


### Supported features and ToDo list
- [x] Multiple GPUs for training

### Requirements:
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04/18.04)
* Python 3.6+
* PyTorch 1.6

### Usage: 

a. Clone the repository.
```shell
git clone --recursive https://github.com/UsamaHasan/AEPI-Automated-Ear-Pinna-Identification
```
```shell
cd AEPI-Automated-Ear-Pinna-Identification && cd src
```
```
python train.py --epochs 100 --batch_size 256 --lr 1e-3
```
### Results:
```
|	Method 	          |		Top 1 Accuracy	|	Top 3 Accuracy	|	
|	---	      |   ---  |	---	|
|	VGG-19	          |		   55.34			  |	    72.13			  |  
|	VGG-19 + SE       |		   59.79			  |	    75.75			  |
|	ResNet-50	        |	   	 60.55			  |	    76.64			  |
|	ResNet-50 + SE    |		   66.22			  |	    81.54			  |
|	ResNet-152+ SE    |		   75.5410		  |	    87.207			|
```
### Dataset:
EarVN1.0: A new large-scale ear images dataset in the wild.
```
Hoang VT. EarVN1.0: A new large-scale ear images dataset in the wild. Data in Brief. 2019 Dec;27:104630. DOI: 10.1016/j.dib.2019.104630.  
```
