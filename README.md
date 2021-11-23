# Cifar100-Classification
My play ground with image classification for Cifar100 dataset

# Usage
- first you need to get the package
````bash
$ git clone https://github.com/dolong2110/Cifar100-Classification.git
````

- Then make sure you are in right directory
````bash
$ cd Cifar100-Classification
````

- Install all the requirement packages
````bash
$ pip install -r requirements.txt
````

- finally train the model
````bash
$ python3 train.py --model resnet18 --image_size 32 --augmentation True
````

Notes here is that:

- `resnet18` is the model's name you can replace its with any available models in my package. For instance resnet152, mobilenet, etc.
- `32` is the images' size. That is the default image size of the cifar100.
- `True` here is whether we should augment data or not.

# Models Using
- self-implement cnn (basic_nn)
- linear regression (linear_regression)
- resnet9 (resnet9)
- resnet18 (resnet18)
- resnet34 (resnet34)
- resnet50 (resnet50)
- resnet101 (resnet101)
- resnet152 (resnet152)
- mobilenet (mobilenet)
- mobilenetv2 (mobilenetv2)

# Report
### 1. Version 1

| Model               | Accuracy |
| ------------------- | -------- |
| `basic_nn`          |          |
| `linear_regression` |          |
| `resnet9`           | 0.6188   |
| `resnet18`          | 0.6405   |
| `resnet34`          | 0.6479   |
| `resnet50`          | 0.6133   |
| `mobilenet`         |          |
| `mobilenetv2`       | 0.4572   |

### 2. Version2
add data augmentation

| Model               | Accuracy |
| ------------------- | -------- |
| `basic_nn`          |          |
| `linear_regression` |          |
| `resnet9`           | 0.6375   |
| `resnet18`          | 0.6739   |
| `resnet34`          |          |
| `resnet50`          |          |
| `mobilenet`         | 0.4635   |
| `mobilenetv2`       |          |
