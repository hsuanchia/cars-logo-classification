# Cars-logo-classification
NCNU 1102 machine learning midterm project 
## Environment
* Cuda 11.2
* Numpy 1.19.5
* Pandas 1.4.1
* Tensorflow 2.5.0
* Opencv 4.5.5.64
* Matplotlib 3.5.1
* Tqdm 4.63.1
## Datasets
### Source
* Photo by myself and my partners
* **Most of them are from internet and the copyright of those picture are not belong to me. Please notice me if you don't want your picture be used for my project**
### Train
* 20 brands
* Each brands has 100 original picture
* Augmentation
    * 90 degree
    * 180 degree
    * 270 degree
* Each brand has total 400 picture
### Test
* Each brand in training set has some picture in test set
* Has 200 picture in total
* Include 10 unknown picture, which doesn't belong to any brand
### Training model
* Execute train.py
    * Call train_model function
    * Choose transfer learning model you want and put it as parameter of train_model
    * Transfer model I used
        * VGG16
        * ResNet50V2
        * DenseNet121
### Run model
* Execute inference.py
    * Set model_path into path of model you want to run
    * Set Test_dir into path of test dataset
    * If want to predict single instance, just call predict_single
* [Models I trained](https://drive.google.com/drive/folders/1O5VvhMDYRv5wSza50__I5JxVQngNxQab?usp=sharing)
## Examples
<img width="877" alt="github ex1" src="https://user-images.githubusercontent.com/59002617/166056603-8342a519-cb27-4ad6-ac14-ee07cd14033d.png">

<img width="880" alt="github ex2" src="https://user-images.githubusercontent.com/59002617/166056713-524681f7-7d89-480c-abd5-4785826034d1.png">

