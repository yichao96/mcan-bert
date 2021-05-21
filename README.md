## MCAN Model with BERT for OK-VQA 
---

This work utilizes bert to encode the question features instead of the LSTM in [MCAN model](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html).
Evaluating the method on [OK-VQA](https://okvqa.allenai.org/index.html) dataset. Besides we use [visual genome dataset](http://visualgenome.org/) to expand the OKVQA train samples. Similar to existing strategies, we preprocessed the samples by two rules:

&emsp;1. Select the QA pairs with the corresponding images appear in the OK-VQA train splits

&emsp;2. Select the QA pairs with the answer appear in the processed answer list (occurs more than 2 times in whole OK-VQA answers).




### Prerequisites
---
&emsp;1. We use devices include: [Cuda 10.1](https://developer.nvidia.com/zh-cn/cuda-toolkit) and [Cudnn](https://developer.nvidia.com/cudnn) 


&emsp;2. First you can create a new envs by conda:

```
    $ conda create -n vqa python==3.6.5
    $ source activate vqa
```


&emsp;3. Install pytorch and torchvision(you can reference the [Pyotrch Install](https://pytorch.org/get-started/previous-versions/)):

```
    $ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

&emsp;4. Install [SpaCy](https://spacy.io/) and initialize the [GloVe](https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz) as follows

```
    $ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
    $ pip install en_vectors_web_lg-2.1.0.tar.gz
```

&emsp;5. Install [Transformers](https://huggingface.co/transformers/installation.html) 
```
    $ pip install transformers==2.11.0 
```

### Dataset Download
---
#### a. Image Download
The image features are extracted using the [bottom-up-attention strategy](https://github.com/peteanderson80/bottom-up-attention), with each image being represented as an dynamic number (from 10 to 100) of 2048-D features.
You can download from this [Bottom-up-Attention](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O)
. Downloaded files contains three files: train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz; The image of the OK-VQA dataset is
a subset of the VQA 2.0 datset.

#### b. Question Download
The question-answer pair you can download in this [OKVQA website](https://okvqa.allenai.org/download.html). Downloaded files contains:
Testing annotations, Testing annotations.

We expand the question sample of training data, so the new training data we provide in the path: /datasets/okvqa/, 
which contrains: OpenEnded_mscoco_train2014_questions_add_VG.json, mscoco_train2014_annotations_add_VG.json


### Training
---

The following script will start training with the default hyperparameters:

```
$ pytho run.py --RUN=train --GPU=0 --MODEL=large
```

All checkpoint files will be saved to:
```
ckpts/ckpt_<VERSION>/epoch<EPOCH_NUMBER>.pkl
```

and the training log file will be placed at:
```
results/log/log_run_<VERSION>.txt
```

You also can set the "EVAL_EVERY_EPOCH=True" in file "cfgs/base_cfgs.py", which can eval in every epoch.


### Val and Test
---

The following script will start valing with the default hyperparameters:
```
$ python run.py RUN=val --GPU=0 --CKPT_PATH=YOUR_CKPT_PATH
```


The following script will start testing with the default hyperparameters:
```
$ python run.py RUN=test --GPU=0 --CKPT_PATH=YOUR_CKPT_PATH
```
You can find the test result in:
```
/results/result_test/result_run_<'VERSION+EPOCH'>.json
```


### Result 
---
In our expriments, the accuracy result on OK-VQA test set as follows:

|  Overall   | VT | BCP |  OMC |  SR| CF| GHLC | PEL | PA | ST | WC | Other|
|  ----  | ----  | ----  |----  |----  |----  |----  |----  |----  |----  |----  |----  |
| 34.80| 30.87| 28.72| 32.01| 45.75| 36.52| 33.05| 31.68|35.05 | 33.33| 47.44 | 31.08| 


---

**Here, we thanks so much these great works:  [mcan-vqa](https://github.com/MILVLG/mcan-vqa), [huggingface](https://github.com/huggingface/transformers) and [OK-VQA](https://okvqa.allenai.org/index.html)** 

































