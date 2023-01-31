# Towards generalizable DeepFake detection with continual learning on limited data
Code used for [Towards generalizable DeepFake detection with continual learning on limited data](http://).



## Updates
- Mar-16-2022: first released


## Introduction
We propose an architecture-based generalization deepfake detection approach that combines spectral analysis and continual learning methods. We prove the proposed generalization approach can perform well when the model is updated with limited new data through extensive experiments on public datasets.

Work architecture:
![avatar](https://github.com/HollyhuangD/deepfake_detection_with_continual_learning_on_limited_data/blob/main/pic/CL-proposal-network.png)



## Set-up
### environment requirements:
python >= 3.6
torch >= 1.1.0
```
pip install -r requirements.txt
```

### prepare data folder
```
mkdir data
```
Dataset are provided ([Google Drive](https://drive.google.com/file/d/1ZagpX2r4cR9exEtNYUQf02WXZhbOLAhq/view?usp=sharing)). Download and upzip to `data` folder.
### prepare model output folder
```
mkdir model
```
Pre-trained models are provided ([Google Drive](https://drive.google.com/file/d/1lUveXB6YgiXGuyRrM8d_B5wAGkWLeORZ/view?usp=sharing)). Download and upzip to `model` folder.

## How to train the model reported in the paper
### Run all in one
```
python ./run_all_train.py
```

### iCarl 
```
python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake --test_set GAN-S VFHQ-F ForenSynths DeepFake NeuralTextures --continual_learning CL --cl_type ni --agent ICARL --retrieve random --update random --mem_size 5000 
```
### GDumb  
```
python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake --test_set GAN-S VFHQ-F ForenSynths DeepFake NeuralTextures --continual_learning CL --cl_type=ni --agent GDUMB --mem_size 3000 --mem_epoch 10 --minlr 0.0005 --clip 10 
```

### LWF
```
python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake --test_set GAN-S VFHQ-F ForenSynths DeepFake NeuralTextures --continual_learning CL --cl_type=ni --agent LWF  
```

### MIR
```
python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake --test_set GAN-S VFHQ-F ForenSynths DeepFake NeuralTextures --continual_learning CL --cl_type=ni --agent ER --retrieve MIR --update random --mem_size 5000
```

## How to test single model
### Run test
You can replace the --loadmodel and --test_set
```
python ./run.py --runtest True --loadmodel './model/VFHQ-F_fft_0_resnet_checkpoint_latest.pth' --test_set VFHQ ForenSynths
```
Pre-trained models are provided ([Google Drive](https://drive.google.com/file/d/1lUveXB6YgiXGuyRrM8d_B5wAGkWLeORZ/view?usp=sharing)). Download and upzip to `model` folder.


## If you need to use code, model or data, please cite the paper "Towards generalizable DeepFake detection with continual learning on limited data".
