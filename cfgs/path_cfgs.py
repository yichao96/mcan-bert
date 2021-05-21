# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = '../datasets/okvqa/'

        # bottom up features root path
        self.FEATURE_PATH = '../datasets/coco_extract/'

        self.PRETRAINED_PATH = '../pretrained/en_bert_base_uncased/'


        self.init_path()


    def init_path(self):

        self.IMG_FEAT_PATH = {
            'train': self.FEATURE_PATH + 'train2014/',
            'val': self.FEATURE_PATH + 'val2014/',
            'test': self.FEATURE_PATH + 'val2014/'
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'OpenEnded_mscoco_train2014_questionsadd_VG.json',
            'val': self.DATASET_PATH + 'OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'OpenEnded_mscoco_val2014_questions.json'
        }

        self.ANSWER_PATH = {
            'train': self.DATASET_PATH + 'mscoco_train2014_annotations_add_VG.json',
            'val': self.DATASET_PATH + 'mscoco_val2014_annotations.json',
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

