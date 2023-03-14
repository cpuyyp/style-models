from nltk.parse.corenlp import CoreNLPParser,CoreNLPDependencyParser
import pandas as pd
import numpy as np
# import collections
from tqdm.auto import tqdm, trange
import textstat
from datasets import load_dataset
from transformers import AutoTokenizer
import spacy_alignments as tokenizations
import string
from wordfreq import word_frequency, zipf_frequency
import re
import os
import json
import ast
import argparse
import gc

# from minicons import scorer

import spacy

# load external models 
nlp = spacy.load('en_core_web_sm') 

with open('/home/jz17d/Desktop/xpos2upos.json') as f:
    xpos2upos = json.load(f)

with open('/home/jz17d/Desktop/dependency2id.json') as f:
    dependency2id = json.load(f)

bert_checkpoint = 'bert-base-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

pos_checkpoint = "/scratch/data_jz17d/result/pos_mlm_corenlp/retrained_all_pos_mlm_22/checkpoint-95000/"
pos_tokenizer = AutoTokenizer.from_pretrained(pos_checkpoint, local_files_only=True)

def clean_text(text):
    # there are some unrecognized character
    text = text.replace('&amp;', '')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('nbsp;', ' ')
    
    text = text.replace('', '')
    text = text.replace('�', '')
    
    # some money and symbles
    text = text.replace('£', ' £ ')
    text = text.replace('$A', 'A$')
    text = text.replace('+', ' + ')
    
    # typos
    text = text.replace("doeen't", "doesn't")
    text = text.replace("M&Ms", "M&M's")
    text = text.replace("G&Ts", "G&T's")
    text = text.replace("HOT&SPiCY", "HOT & SPiCY")
    
    # dirty words
    text = text.replace("F**K", "FUCK")
    text = text.replace("f**k", "fuck")
    text = text.replace("b*tch", "bitch")
    
    # question numbering
    text = re.sub(r'\s?([0-9]{1,2}[\.\:])([0-9]{1,2}[ABCDabcd])\s?', r" \g<1> \g<2> ", text)
    
    # special cases where corenlp and bert deal differently
    # compact timestamp, eg. xx:xxam
    text = re.sub(r'\s?(\d{0,2}[.:]?\d{0,2}[.:]?\d{1,2})([apAP][Mm])\s?', r' \g<1> \g<2> ', text)
    # numbers with commas, e.g. xx,xxx,xxx
    text = re.sub(r'\s?([0-9]+)?(,)?([0-9]+)(,)([0-9]+)\s?', r' \g<1>\g<3>\g<5> ', text)
    # xxxth
    text = re.sub(r'\s?([0-9]+)(st|nd|rd|th)\s?', r' \g<1>\g<2> ', text)
    # x/x
    text = re.sub(r'\s?([0-9]+)/([0-9]+)\s?', r' \g<1>/\g<2> ', text)
    # seconds (maybe applies to other time units as well)
    text = re.sub(r'\s?([0-9.]+)([Ss])\s?', r' \g<1> \g<2> ', text)
    
    # slangs
    text = re.sub(r'\s?([Gg][Oo][Nn])([Nn][Aa])\s?', r' \g<1> \g<2> ', text)
    text = re.sub(r'\s?([Gg][Oo][Tt])([Tt][Aa])\s?', r' \g<1> \g<2> ', text)
    text = re.sub(r'\s?([Ww][Aa][Nn])([Nn][Aa])\s?', r' \g<1> \g<2> ', text)
    text = re.sub(r'\s?([Ll][Ee][Mm])([Mm][Ee])\s?', r' \g<1> \g<2> ', text)
    text = re.sub(r'\s?([Gg][Ii][Mm])([Mm][Ee])\s?', r' \g<1> \g<2> ', text)
    text = re.sub(r'\s?([Dd])([Uu][Nn][Nn][Oo])\s?', r" \g<1>on't know ", text)
    text = re.sub(r'\s?([Cc])\'[Mm][Oo][Nn]\s?', r" \g<1>ome on ", text)
    text = re.sub(r'\s?[\'’`´][Tt]is\s?', r" it is ", text)
    
    # n't
    text = re.sub(r'\s?([Dd][Oo]|[Dd][Oo][Ee][Ss]|[Dd][Ii][Dd]|[Aa][Ii]|[Ii][Ss]|[Aa][Rr][Ee]|[Ww][Aa][Ss]|[Ww][Ee][Rr][Ee]|[Cc][Aa]|[Ww][Oo]|[Ww][Oo][Uu][Ll][Dd]|[Cc][Oo][Uu][Ll][Dd]|[Hh][Aa][Vv][Ee]|[Hh][Aa][Ss]|[Hh][Aa][Dd]|[Ss][Hh][Oo][Uu][Ll][Dd])([Nn])[\'’`´]?([Tt])\s?', r" \g<1> \g<2>'\g<3> ", text)
    text = re.sub(r'\s?(n[\'’`´]t|N[\'’`´]T)\s?', r' \g<1> ', text)
    text = re.sub(r'\s?(can|CAN|Can)(not|NOT)\s?', r' \g<1> \g<2> ', text)

    return text

def align(text):
    # corenlp_tokens = list(tokenizer.tokenize(text)) # do not need an extra port to tokenize
    result = depparser.api_call(text, 
                            properties={"annotators": "ssplit,tokenize",
                                        "ssplit.eolonly": "ture",
                                        "tokenize.strictAcronym": "true"
                                        })
    corenlp_tokens = [token['originalText'] for sent in result['sentences'] for token in sent['tokens'] ]

    bert_tokens = bert_tokenizer.tokenize(text)
    
    a2b, b2a = tokenizations.get_alignments(corenlp_tokens, bert_tokens)
    if '[UNK]' in bert_tokens:
        last = -1
        for i in range(len(b2a)):
            if b2a[i] == []:
                b2a[i] = [last+1]
            last = b2a[i][-1]
    alignment = [item for sublist in b2a for item in sublist]
    if len(alignment) != len(bert_tokens):
        print(f'{len(alignment)} != {len(bert_tokens)} length mismatch!!!')
        print(text)
        if '[UNK]' in bert_tokens:
            print('found [UNK] and cannot fix!!!')
    return alignment, len(alignment) != len(bert_tokens)

def get_word_freq_category(word):
    # top zipf freq is 7.73 for word "the"
    # so normally the range for zipf freq is [0,8)
    # for punctuations, return 9
    if word in string.punctuation:
        return 9
    freq = zipf_frequency(word, lang='en')
    return 0 if freq == 0 else int(freq+1)

def convert_pos_seq(pos_seq):
    upos_seq = []
    for token in pos_seq:
        upos_seq.append(xpos2upos[token])
    return upos_seq   

def parse_dependency(text):
    parsed = depparser.raw_parse(text) # this not only parse the tree, but also tokenize the text
    conll_dep = next(parsed).to_conll(4)
    lines = conll_dep.split('\n')
    
    edge_index = []
    hetoro_edge = []
    pos = []
    num_syllable = []
    num_char = []
    word_freq = []
    for i,line in enumerate(lines[:-1]):
        l = line.split('\t')
        edge_index.append([i+1, int(l[2])])
        hetoro_edge.append(l[3])
        pos.append(l[1])
        num_syllable.append(textstat.syllable_count(l[0]))
        num_char.append(len(l[0]))
        word_freq.append(get_word_freq_category(l[0]))
    upos = convert_pos_seq(pos)
    return edge_index, hetoro_edge, pos, upos, num_syllable, num_char, word_freq
    
def process_features(df):
    texts = df['text']
    
    edge_indexs = []
    hetoro_edges = []
    pos_seqs = []
    upos_seqs = []
    num_syllables = []
    num_chars = []
    word_freqs = []

    alignments = []
    mismatched = []
    for text in tqdm(texts):
#         text = clean_text(text) # text are cleaned before this

        edge_index, hetoro_edge, pos, upos, num_syllable, num_char, word_freq = parse_dependency(text)
        edge_indexs.append(edge_index)
        hetoro_edges.append(hetoro_edge)
        pos_seqs.append(pos)
        upos_seqs.append(upos)
        num_syllables.append(num_syllable)
        num_chars.append(num_char)
        word_freqs.append(word_freq)

        alignment, mismatch_found = align(text)
        alignments.append(alignment)
        if mismatch_found:
            mismatched.append(text)
    
    df['edge_indexs'] = edge_indexs
    df['hetoro_edges'] = hetoro_edges
    df['pos_seqs'] = pos_seqs
    df['upos_seqs'] = upos_seqs
    df['num_syllables'] = num_syllables
    df['num_chars'] = num_chars
    df['word_freqs'] = word_freqs

    df['alignments'] = alignments
    
    return df, mismatched

def split_sentence(raw_text, processer='spacy'):
    if processer == 'spacy':
        doc = nlp(raw_text)
        sentences = [sent.text.strip() for sent in doc.sents]

    elif processer == 'corenlp':
        result = depparser.api_call(raw_text, 
                                    properties={"annotators": "tokenize,ssplit",
                                                "ssplit.eolonly": "false",
                                                "tokenize.strictAcronym": "true"
                                                })
        sentences = [' '.join([token['originalText'] for token in sent['tokens']]) for sent in result['sentences']]

    return sentences

def split_sentences(df):
    '''
    apply split_sentence to all document
    '''
    df['text'] = df['text'].apply(split_sentence)
    return df

def load_raw_csv_and_explode(config, split):
    
    raw_dir = config['raw_dir']
    filename = config[f'{split}_set']
    need_ssplit = config['need_ssplit']

    df = pd.read_csv(f'{raw_dir}/{filename}')
    if 'doc_id' not in df.columns:
        df = df.reset_index().rename({'index':'doc_id'}, axis=1)

    if need_ssplit:
        df = split_sentences(df)
    else:
        df['text'] = df['text'].str.strip('\n').str.split('\n')
    df = df.explode('text')
    return df

def process_doc(df, all_features):
    '''
    all_features include all cols except 'text'
    '''
    
    agg = {col: lambda x: x.tolist() for col in ['text']+all_features}

    return df.groupby(['doc_id','author_id']).agg(agg).reset_index()

def process_segments(df, window_size, all_features):
    '''
    window_size should be >= 2
    all_features include all cols except 'text'
    '''
    def agg_f(x):
        if x.shape[0] < window_size: # if the entire doc has less sentences than the window size, it will be kept as a single segment.
            return [x.tolist()]
        else:
            return [win.tolist() for win in x.rolling(window_size) if win.shape[0] == window_size] 
    agg = {col: agg_f for col in ['text']+all_features}

    return df.groupby(['doc_id','author_id']).agg(agg).explode(['text']+all_features).reset_index()

# to deal with cases like int64
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def split_individuals(df_agg, outfolder):
    os.makedirs(outfolder, exist_ok = True) 
    count = 0
    for i in trange(len(df_agg), leave=False):
        data = df_agg.iloc[i].to_dict()
        with open(f'{outfolder}/{count}.json', 'w') as f:
            json.dump(data, f, cls=NpEncoder)
        count += 1

# global settings
all_features = ['edge_indexs', 'hetoro_edges', 'pos_seqs', 'upos_seqs', 'num_syllables', 'num_chars', 'word_freqs', 'alignments', ] # col 'text' is not included

dataset_configs = {}
dataset_configs['ccat50'] = {'raw_dir': '/home/jz17d/Desktop/style-models/data/CCAT50/reprocessed',
                             'scratch_dir': '/scratch/data_jz17d/data/ccat50',
                             'splits': ['train', 'test'],
                             'train_set': 'CCAT50_train.csv',
                            #  'val_set': 'CCAT50_AA_val.csv', # resplit ccat do not have val
                             'test_set': 'CCAT50_test.csv',
                             'need_ssplit': False,
                             'num_author': 50,
                             }

dataset_configs['imdb62'] = {'raw_dir': '/home/jz17d/Desktop/style-models/data/imdb/processed',
                             'scratch_dir': '/scratch/data_jz17d/data/imdb62',
                             'splits': ['train', 'val', 'test'],
                             'train_set': 'imdb62_train.csv',
                             'val_set': 'imdb62_AA_val.csv',
                             'test_set': 'imdb62_AA_test.csv',
                             'need_ssplit': True,
                             'num_author': 62,
                             }

dataset_configs['blogs50'] = {'raw_dir': '/home/jz17d/Desktop/style-models/data/blogs50/processed',
                              'scratch_dir': '/scratch/data_jz17d/data/blogs50',
                              'splits': ['train', 'val', 'test'],
                              'train_set': 'blogs50_train.csv',
                              'val_set': 'blogs50_AA_val.csv',
                              'test_set': 'blogs50_AA_test.csv',
                              'need_ssplit': True,
                              'num_author': 50,
                              }

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Select dataset, preprocess features, split into individual files')

    parser.add_argument('--dataset', type=str, required=True, choices=['ccat50', 'imdb62', 'blogs50'])
    parser.add_argument('--splits', type=str, nargs='+', default=None)
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--save_processed_csv', type=bool, default=True)
    parser.add_argument('--save_seg_csv', type=bool, default=False)
    parser.add_argument('--create_doc_level', type=bool, default=True)
    parser.add_argument('--force_reprocessed', type=bool, default=False)

    parser.add_argument('--corenlp_port', type=int, default=9000)

    args = parser.parse_args()
    depparser = CoreNLPDependencyParser(url=f'http://localhost:{args.corenlp_port}')

    dataset = args.dataset
    config = dataset_configs[dataset]

    splits = args.splits if args.splits is not None else config['splits']
    window_sizes = args.window_sizes
    save_processed_csv = args.save_processed_csv
    save_seg_csv = args.save_seg_csv
    create_doc_level = args.create_doc_level
    force_reprocessed = args.force_reprocessed
    
    scratch_dir = config['scratch_dir']

    for split in splits:
        file = f"{scratch_dir}/processed_{split}.csv"
        # if processe file found, load it and skip feature processing
        if os.path.isfile(file) and not force_reprocessed:
            df = pd.read_csv(file)
            for col in all_features: # in the processed file, 'text' col is just simple string. No need to eval.
                df[col] = df[col].apply(ast.literal_eval)
                gc.collect() # may release some memory
            print('found existing processed file, skip feature processing!')
        else:
            # load and split into sentences
            df = load_raw_csv_and_explode(config, split)

            # clean
            df['text'] = df['text'].apply(clean_text).str.strip().replace('', np.nan)
            df = df.dropna(subset=['text'])

            # process features
            df, mismatched = process_features(df) # usually encounter StopIteration. might exist empty string
            if save_processed_csv:
                df.to_csv(f"{scratch_dir}/processed_{split}.csv", index=False)
            if mismatched:
                with open(f"{scratch_dir}/{split}_mismatched.txt",'w') as f:
                    f.write('\n'.join(mismatched))
        
        if split == 'test':
            # these two files are used in the evaluation
            test_docid2index = {int(doc_id):i for i,doc_id in enumerate(sorted(df['doc_id'].unique()))}
            with open(f'{scratch_dir}/test_docid2index.json', 'w') as f:
                json.dump(test_docid2index, f, cls=NpEncoder)
            doc_true = df[['doc_id', 'author_id']].value_counts().reset_index().sort_values('doc_id')['author_id'].values
            np.save(f'{scratch_dir}/doc_true.npy', doc_true)

        # collect to form segments
        for window_size in window_sizes:
            df_agg = process_segments(df, window_size = window_size, all_features=all_features)
            outfolder = f'{scratch_dir}/segment_{window_size}_{split}/raw'
            split_individuals(df_agg, outfolder)
            if save_seg_csv:
                file_name = f"sent_{window_size}_{split}"
                df_agg.to_csv(f"{scratch_dir}/{file_name}.csv", index=False)    
        
        # collect to form docs
        if create_doc_level:
            df_agg = process_doc(df, all_features=all_features)
            outfolder = f'{scratch_dir}/doc_{split}/raw'
            split_individuals(df_agg, outfolder)
            if save_seg_csv:
                file_name = f"doc_{split}"
                df_agg.to_csv(f"{scratch_dir}/{file_name}.csv", index=False)

    print('finished!')

        
