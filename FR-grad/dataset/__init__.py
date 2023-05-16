from dataset.cfp import CFPLoader
from dataset.ytf import YTFLoader
from dataset.lfw import LFWLoader
from dataset.casia import CASIALoader
LOADER_DICT = {
    'lfw': LFWLoader,
    'ytf': YTFLoader,
    'cfp': CFPLoader,
    'casia':CASIALoader
}