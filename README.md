# Grounded-Minimal-Edit
Code for EMNLP 2021 paper "Transferable Persona-Grounded Dialogues via Grounded Minimal Edits"

### Dependencies

```shell
pip install transformers==2.3.0 torch==1.2.0 nltk matplotlib tensorboardX
```
Download the nltk stopwords.
```python
import nltk
nltk.download('stopwords')
```
Download persona evaluator from [here](https://drive.google.com/file/d/1VJ_TM6sy556h-QAmKWOgPEI5L1tN0yno/view?usp=sharing) as `dnli-bert/dnli_model.bin`.

### PersonaMinEdit Dataset Format
Training data (data/personachat-ucpt/train.json):
```
[
    ...
    {
        "context": tuple of strs,
        "response": str,
        "persona": str or an empty list,
    },
    ...
]
```
Validation and test data (data/personachat-ucpt/{valid, test}.json):
```
[
    ...
    {
        "context": tuple of strs,
        "original_response": str,
        "intervening_persona": tuple of strs,
        "references": tuple of strs,
    },
    ...
]
```

### Grounded Minimal Editing Experiment
Use `GME` as the root directory. 

#### Train
1. Train the editor model.
```
python3 train.py 
```

#### Test
1. Download checkpoint (seed=0) from [here](https://drive.google.com/file/d/1qlUJzc6kF3x_iDBhpXkX5GUlh9Sj35VT/view?usp=sharing) as `outputs/saved_model/persona-chat-cprm-smooth_eps0.1-grad_thres3-tau3-0/best-model.ckpt`. This step is not necessary if you do the training above. 
```
python3 test.py 
```

### Transfer Learning Experiment
Use `GME-Zero-Shot` as the root directory. 

1. Download checkpoint (seed=0) from [here](https://drive.google.com/file/d/1qlUJzc6kF3x_iDBhpXkX5GUlh9Sj35VT/view?usp=sharing) as `outputs/saved_model/persona-chat-cprm-smooth_eps0.1-grad_thres3-tau3-0/best-model.ckpt`. If you do the training of the Grounded Minimal Editing experiment, copy the saved checkpoint as `outputs/saved_model/persona-chat-cprm-smooth_eps0.1-grad_thres3-tau3-0/best-model.ckpt`. 
```
python3 test.py 
```