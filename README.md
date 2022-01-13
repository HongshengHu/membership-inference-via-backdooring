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
python train_clean.py --gpu-id 0 --checkpoint 'checkpoint/benign_model'
```
## Train a backdoored model
* One data owner's data was collected and used: the default trigger pattern is a 3x3 white square and stamped in the bottom right of the samples. You can vary different --y_target, --trigger_size, and --marking_ratio to see how these factors affact the backdoor attack success rate. Note that adjusting the coordinate of the trigger.
```python
python train_MIB.py --gpu-id 0 --checkpoint 'checkpoint/one_owner' --trigger 'white_square' --y_target 1 --trigger_size 3 --trigger_coordinate_x 29 --trigger_coordinate_y 29 --marking_rate 0.001
```

* Multiple data owner's data was collected and used: You can vary the number of data owners by changing --num_users. In the experiments, each data owner uses a different thrigger pattern and a different target label.
```python
python train_MIB_multi.py --gpu-id 0 --checkpoint 'checkpoint/multi_owner' --num_users 10
```
