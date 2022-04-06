# Towards generalizable DeepFake detection with continual learning on limited data
Code used for [Towards generalizable DeepFake detection with continual learning on limited data](http://).



## Updates
- Mar-16-2022: first released


## Introduction
We propose an architecture-based generalization deepfake detection approach that combines spectral analysis and continual learning methods. We prove the proposed generalization approach can perform well when the model is updated with limited new data through extensive experiments on public datasets.

Work architecture:
![avatar](https://lh3.googleusercontent.com/XLpuoZ0z78sI2m1r54aDmv80r7_2bXSdnRhzbQFPLFZWEMRP0054ZhKuiCS9bjEHpF3rZ5rxA9y07cTYK1plSA97z5YRv1npVK9vM0uMUFuLviXduew_FgKz-ewD-sFK7dvTxcXTnwcRvTth5l1tNVv1N5UmIynQpGOQDxdjqoUD11TZ3cJZHaE-PVwCBJXq0mKPsKgLSWqKilj5Cgnc0X_Mf0kY8GEoB3O4N6rY8fD-8B6WOusOOpA4T2chkogOkl5573GZeKc76Torou-ggcK28y_7rc70fZjOftqTS4KWg2C8Lc4BtNl3OUzzUCBF12xVrB9Sptyo4Hg2z1q_HoZoQBHpJk9Oebuta1T-HRZ09vINPWJWSC4qc4Nc5tI2ehwS3bL26t05hUnfarnyWZO-yp6St1HSw3IeAjs85wLm47FZu9wu8FoKAyqdSH07hV6v3eLMobLQpPlhzm0LfVxcY1CWUuTefNlZRXsmYB4F3lqsM9htn2ByiLbC8jFCYnayrJi-lMUAawZkga2WZ3v7q77nv6tbyjQ7HtM9OXpB912jY3dsV_qImXJpiJ0iEUSIAStyQDcU7Ex_DvlB2ix2-UcLCr_JRTHxMwQit-WyJuTd4ZKrjYBEHcSLYnGjzodAWkEYI0YjYBn5M1p1sfkY70RUkYjeLCMByt4UBwaVWZZhDKS8NuHSyg1kQs0dEUAFS_fQrsOGWFcEBoGnRXkGSKVQnLXz9xQLlhEdZjDVAfbfyr1muwPCr02Y2uzg4bHTDmZ4qVDwv6H9Qk8_tsDNuuOuC0lUY5g=w2402-h866-no?authuser=0)



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
