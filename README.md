# Style models
## Introduction
This repository is majorly for syntactic representation learning codes and related stuffs. However, other earlier experiments are also included.

## Dataset Access

Download them seperately before training. Because some processed datasets are extremely large, we are not providing the processed version; instead, we are sharing the raw data.

- Bookcorpus: This dataset is used for pre-training POS BERT. Download it from Huggingface https://huggingface.co/datasets/bookcorpus with
```python
from datasets import load_dataset
dataset = load_dataset("bookcorpus")
```

- CCAT50: CCAT50 is a popular benchmark for authorship attribution or authorship verification.
    - Raw: the only difference with the original version is that I added a column "doc_id" for performning majority vote. https://drive.google.com/file/d/1VYhmFmCDBxP65HJvc9t-5_BheRSnW3Cm/view?usp=sharing
    - Processed: Added features including POS, UPOS (a simplified POS), edge_indexs (edge list), hetoro_edges (dependency types), num_syllables, num_chars, word_freqs, and alignments (to match POS and subwords). https://drive.google.com/file/d/1MLv8imd9jdvr6IDCJzz7OYfCHHuQmoKZ/view?usp=sharing
- Blogs50: Download here: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm . We use https://github.com/JacobTyo/Valla/blob/main/valla/dsets/blogs.py (although it mentions to download from Huggingface,) to have a first round process and continue build on it. 
- CMCC: Email authoros of: Creating and Using a Correlated Corpora to Glean Communicative Commonalities http://www.lrec-conf.org/proceedings/lrec2008/pdf/771_paper.pdf. Then run a first round process with https://github.com/JacobTyo/Valla/blob/main/valla/dsets/CMCC.py

## Preprocessing
The preprocessing includes three steps. 
1. POS parsing
2. Other syntax feature collecting
3. Data augmentation

### POS parsing

We conduct the POS parsing with the most up to date Stanford CoreNLP. (Current version 4.5.2 by July 5, 2023)

1. Download the newest version from https://stanfordnlp.github.io/CoreNLP/.

2. Following the guide in https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK, change to the corresponding directory, start the CoreNLP server running in the background using 
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 &
```
The server will load selected annotators by the -preload flag. For a complete list of annotators, check https://stanfordnlp.github.io/CoreNLP/annotators.html 

3. (optional) For alignment purposes, I found an issue that CoreNLP will add a period at the end of some abbreviations even if they are not in the original text, as described in https://github.com/stanfordnlp/CoreNLP/issues/1338. Note that this is only used to tokenize. Downstream pipelines may have wrong results without the trailing period. 
Start another server using
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize" -ssplit.verbose True -tokenize.options strictAcronym=true -status_port 9001 -port 9001 -timeout 15000 &
```
Note that it is also possible to pass the configuration "strictAcronym=true" in the python interface. For example, 

```python
result = depparser.api_call(test, 
                            properties={"annotators": "tokenize,ssplit",
                                        "ssplit.eolonly": "false",
                                        "strictAcronym": "true"
                                        })
```

4. Then in Python, the parser can be loaded with
```python
from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
parser = CoreNLPParser(url='http://localhost:9001', tagtype='pos')
depparser = CoreNLPDependencyParser(url='http://localhost:9000')
```

Optional preprocessings
POS to UPOS:

The CoreNLP POS parser only provides XPOS annotation. To get UPOS, the universal dependency organization provides a simple conversion table here: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html. We put the conversion in file xpos2upos.json.

For better conversion, CoreNLP provides a converter but it’s not accessible in Python, only with java. Following instructions here: https://github.com/clulab/processors/wiki/Converting-from-Penn-Treebank-to-Basic-Stanford-Dependencies. 

### Other syntax feature collecting

- Num_chars: this is simply ```len(word)```. For special token "CLS" etc, it will be 0.
- Num_syllables: we use "textstat" which internally uses "Pyphen". For special token "CLS" etc, it will be 17 (which is more than the maximum syllables that a normal word can have. However, for some chemical name, it can be super long, which is not considered for now). For punctuations, it will be 0.
- word_freq: we use the function ```zipf_frequency()``` in the "wordfreq" package. For punctuations, it will be 9. For special token "CLS" etc, it will be 10. (```zipf_frequency()``` measures the log10 of frequency per billion words. So the maximum would never exceed 8.)

### Data augmentation

Applying Sliding Over Sentences (SOS) on a given dataset. The dataset needs to include an index for document, "doc_id" by default.



### Put together

"preprocess.py" includes all necessary codes for preprocessing step 4 to data augmentation. By passing arguments, you can get a training-ready, augmented version with a single command. An example of running the code is shown here:

```python
python preprocess.py --dataset ccat50 --window_sizes 2,3,4
```

## Other experiments

### Multitask style
- Datasets: PASTEL and xSLUE. Can be downloaded here: https://github.com/dykang/PASTEL and https://github.com/dykang/xslue. 

- multitask_style_learning_utils.ipynb: All definitions and functions are defined here. To load this from another notebook, simply run ```%run ./multitask_style_learning_utils.ipynb```

To train a multitask model, a simple demo here
```python
# select tasks from 22 datasets
selected_tasks = ['PASTEL_country', 
                  'SARC', 
                  'ShortHumor', 
                 ] 

# put all args into this training_args
training_args = TrainingArgs(selected_tasks=selected_tasks,
                             # select from 'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', etc.
                             base_model_name='bert-base-uncased',
                             # True: freeze all. Or input an int to freeze first n attention layers.
                             freeze_bert=False, 
                             # use pooler output or the CLS token embedding
                             use_pooler=True, 
                             num_epoch=5,
                             # drop rows that exceed this limit for each dataset 
                             data_limit=30000, 
                            ) # there are some other args, check TrainingArgs definition

# training and evaluating
model = init_model(training_args) # initial model and tokenizer
freeze_model(model, training_args.freeze_bert) # freeze model layers
df_evaluation, df_loss_per_step, model = train_model(model, training_args) # train

# run bertology
eval_dataloader = MultiTaskTestDataLoader(training_args, split='dev') 
attn_entropy, head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, training_args)

# visualize bertology result
imshow(attn_entropy)
imshow(head_importance)
```
- other notebooks named by style1+style2+...: These notebooks contains the scripts and output of training one or more specific model(s).
