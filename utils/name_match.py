
from agents.exp_replay import ExperienceReplay
from agents.gdumb import Gdumb
from agents.icarl import Icarl
from agents.lwf import Lwf
from continuum.dataset_scripts.unidatasets import GAN_S, VFHQ, ForenSynths, FaceForensics, GAN_S_LD, \
    VFHQF_LD, ForenSynths_LD, FaceForensics_LD, DeepFake, NeuralTextures, UniDataset1, UniDataset2, UniDataset3

from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve

data_objects = {
    'VFHQ': VFHQ,
    'ForenSynths': ForenSynths,
    'GAN-S': GAN_S,
    'FaceForensics': FaceForensics,
    'VFHQ-F-LD': VFHQF_LD,
    'ForenSynths-LD': ForenSynths_LD,
    'GAN-S-LD': GAN_S_LD,
    'FaceForensics-LD': FaceForensics_LD,
    'DeepFake': DeepFake,
    'NeuralTextures': NeuralTextures,
    'UniDataset1': UniDataset1,
    'UniDataset2': UniDataset2,
    'UniDataset3': UniDataset3
}

agents = {
    'ER': ExperienceReplay,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve

}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update
}
