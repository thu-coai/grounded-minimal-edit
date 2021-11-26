# Unsupervised-Persona-Transfer

### Dependencies

```shell
pip install transformers==2.3.0 torch==1.2.0 nltk matplotlib tensorboardX
```
Download the nltk stopwords.
```python
import nltk
nltk.download('stopwords')
```
Download persona evaluator from [here]() as `dnli-bert/dnli_model.bin`.

### PersonaMinEdit Dataset Format
Training data (data/personachat-ucpt/train.json):
```json
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
```json
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
Download checkpoint (seed=0) from [here]() as `GME/outputs/saved_model/persona-chat-cprm-smooth_eps0.1-grad_thres3-tau3-0/best-model.ckpt`.
```
cd GME
python3 test.py 
```

### Transfer Learning Experiment
Download checkpoint (seed=0) from [here]() as `GME-Zero-Shot/outputs/saved_model/persona-chat-cprm-smooth_eps0.1-grad_thres3-tau3-0/best-model.ckpt`.
```
cd GME-Zero-Shot
python3 test.py 
```