# membership-inference-via-backdooring
This repository contains the source code of the paper "Membership Inference via Backdooring for Machine Unlearning".

# Requirement 
* torch==1.8.1
* numpy==1.18.1
* torchvision==0.9.1

The experiments are evaluated on one image dataset of CIFAR-10 and two binary datasets of Location-30 and Purchase-100.



# Experiments on CIFAR-10 datasets
## Train a clean model
```python
python train_standard.py --gpu-id 0 --checkpoint 'checkpoint/benign_model'
```
