from continuum.dataset_scripts.GANDataset_base import GANDataset_base

class GAN_S(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'GAN-S'
        super(GAN_S, self).__init__(scenario, params, name=dataset)

class VFHQ(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'VFHQ'
        super(VFHQ, self).__init__(scenario, params, name=dataset)

class ForenSynths(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'ForenSynths'
        super(ForenSynths, self).__init__(scenario, params, name=dataset)

class FaceForensics(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'FaceForensics'
        super(FaceForensics, self).__init__(scenario, params, name=dataset)

class GAN_S_LD(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'GAN-S-LD'
        super(GAN_S_LD, self).__init__(scenario, params, name=dataset)

class VFHQF_LD(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'VFHQ-F-LD'
        super(VFHQF_LD, self).__init__(scenario, params, name=dataset)

class ForenSynths_LD(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'ForenSynths-LD'
        super(ForenSynths_LD, self).__init__(scenario, params, name=dataset)

class FaceForensics_LD(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'FaceForensics-LD'
        super(FaceForensics_LD, self).__init__(scenario, params, name=dataset)

class DeepFake(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'DeepFake'
        super(DeepFake, self).__init__(scenario, params, name=dataset)


class NeuralTextures(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'NeuralTextures'
        super(NeuralTextures, self).__init__(scenario, params, name=dataset)

class UniDataset1(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'UniDataset1'
        super(UniDataset1, self).__init__(scenario, params, name=dataset)


class UniDataset2(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'UniDataset2'
        super(UniDataset2, self).__init__(scenario, params, name=dataset)


class UniDataset3(GANDataset_base):
    def __init__(self, scenario, params):
        dataset = 'UniDataset3'
        super(UniDataset3, self).__init__(scenario, params, name=dataset)
