import os
ORIGIN_DIR = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/input/origin/'
ANNOTATION_DIR = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/annotation/'

TRAIN_SAMPLE_PATH = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/output/dataset/EDA_CHD/train_sample.txt'
TEST_SAMPLE_PATH = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/output/dataset/EDA_CHD/test_sample.txt'

VOCAB_PATH = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/output/dataset/EDA_CHD/vocab.txt'
LABEL_PATH = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/output/dataset/EDA_CHD/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
TARGET_SIZE = 31
LR = 1e-5
EPOCH = 50

MODEL_DIR = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/output/model/EDA_CHD_model/'
LOG_PATH = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/Logging/EDA_CHD_training.log'

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Bert 
BERT_MODEL = '/home/deng/Maping/EMR-github-projects/FlatNER/BERT_BiLSTM_CRF/HuggingFace_Model/bert-base-chinese'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512