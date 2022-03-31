import torch
from torchvision import transforms

default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}

input_size_match = {
    'VFHQ': [3, 224, 224],
    'ForenSynths': [3, 224, 224],
    'GAN-S': [3, 224, 224],
    'FaceForensics': [3, 224, 224],
    'VFHQ-F-LD': [3, 224, 224],
    'ForenSynths-LD': [3, 224, 224],
    'GAN-S-LD': [3, 224, 224],
    'FaceForensics-LD': [3, 224, 224],
    'DeepFake': [3, 224, 224],
    'NeuralTextures': [3, 224, 224],
    'UniDataset1': [3, 224, 224],
    'UniDataset2': [3, 224, 224],
    'UniDataset3': [3, 224, 224],
}

n_classes = {
    'VFHQ': 2,
    'ForenSynths': 2,
    'GAN-S': 2,
    'FaceForensics': 2,
    'VFHQ-F-LD': 2,
    'ForenSynths-LD': 2,
    'GAN-S-LD': 2,
    'FaceForensics-LD': 2,
    'DeepFake': 2,
    'NeuralTextures': 2,
    'UniDataset1': 2,
    'UniDataset2': 2,
    'UniDataset3': 2,
}

transforms_match = {
    'VFHQ': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'ForenSynths': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'GAN-S': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'FaceForensics': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'VFHQ-F-LD': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'ForenSynths-LD': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'GAN-S-LD': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'FaceForensics-LD': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'DeepFake': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'NeuralTextures': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'UniDataset1': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'UniDataset2': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'UniDataset3': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}



def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
