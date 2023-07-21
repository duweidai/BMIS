# BMIS

A novel backbone for 2D medical image segmentation: smaller, faster, and stronger
skin lesion segmentation
We take skin disease segmentation as an example to introduce the use of our model.
Data preparation
resize datasets (ISIC2018 and PH2) to 224*224 and saved them in npy format.

python data_preprocess.py
Train and Test
Our method is easy to train and test, just need to run "train_and_test_isic.py".

python train_and_test_isic.py
