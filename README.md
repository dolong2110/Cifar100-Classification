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
````
$ python3 train.py --model resnet18 --image_size 32
````

Notes here is that:

- `resnet18` is the model's name you can replace its with any available models in my package. For instance resnet152, mobilenet, etc.
- `32` is the images' size. That is the default image size of the cifar100/