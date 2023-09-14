# Neural Architecture Search for Pneumonia Diagnosis
Code for our paper [Neural architecture search for pneumonia diagnosis from chest X-rays](https://www.nature.com/articles/s41598-022-15341-0). 

## Abstract

Pneumonia is one of the diseases that causes the most fatalities worldwide, especially in children. Recently, pneumonia-caused deaths have increased dramatically due to the novel Coronavirus global pandemic. Chest X-ray (CXR) images are one of the most readily available and common imaging modality for the detection and identification of pneumonia. However, the detection of pneumonia from chest radiography is a difficult task even for experienced radiologists. Artificial Intelligence (AI) based systems have great potential in assisting in quick and accurate diagnosis of pneumonia from chest X-rays. The aim of this study is to develop a Neural Architecture Search (NAS) method to find the best convolutional architecture capable of detecting pneumonia from chest X-rays. We propose a Learning by Teaching framework inspired by the teaching-driven learning methodology from humans, and conduct experiments on a pneumonia chest X-ray dataset with over 5000 images. Our proposed method yields an area under ROC curve (AUC) of 97.6% for pneumonia detection, which improves upon previous NAS methods by 5.1% (absolute).


## How to run code

- Step 1: Run DHE.py on the dataset to generate augmented training dataset. ` run DHE.py `
- Step 2: Perform architecture search ` run arch_search.py `
- Step 3: Train on downstream task i.e Pneumonia detection ` run train.py `

## Dataset

All experiments are carried out using the publicly available [chest X-ray images (with pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle. 



## Citation

Gupta, A., Sheth, P. & Xie, P. Neural architecture search for pneumonia diagnosis from chest X-rays. Sci Rep 12, 11309 (2022). https://doi.org/10.1038/s41598-022-15341-0
