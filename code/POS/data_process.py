import pandas as pd
import numpy as np
from tqdm.auto import trange, tqdm
from nltk.parse.corenlp import CoreNLPDependencyParser
import argparse


def create_dataset(df, num_authors_to_pick = None, picked_author_ids = None, num_sent_per_text = None, save_folder = None, train=True):
    assert bool(picked_author_ids) != bool(unique_authors), "don't give picked_author_ids and unique_authors togethor"
    unique_authors = list(df_ccat['author_id'].unique())
    if not picked_author_ids:
        picked_author_ids = sorted(np.random.choice(unique_authors, replace=False, size=num_authors_to_pick).tolist())
    authors = []
    texts = []
    for author in picked_author_ids:
        df_temp = df[df['author_id'] == author]
        for i_doc in range(len(df_temp)):
            doc = df_temp['text'].iloc[i_doc].split('\n')
            for i in range(len(doc)):
                doc[i] = doc[i].strip()
            doc.remove('')
            for i in range(len(doc)-num_sent_per_text):
                authors.append(author)
                texts.append(' '.join(doc[i:i+num_sent_per_text]))
    df = pd.DataFrame({'author':authors, 'text':texts})
    if save_folder:
        str_author = ','.join(map(str, picked_author_ids))
        file_name = f"author_{str_author}_sent_{num_sent_per_text}_{'train' if train else 'val'}.csv"
        df.to_csv(f"{save_folder}/{file_name}", index=False)
        return df, file_name
    return df

def get_dep_edges(texts):
    homo_edges = []
    pos_seqs = []
    edge_types = []
    for text in tqdm(texts):
        parsed = depparser.raw_parse(text)
        conll_dep = next(parsed).to_conll(4)
        lines = conll_dep.split('\n')
        homo_edge = []
        pos_seq = []
        edge_type = []
        for i,line in enumerate(lines[:-1]):
            l = line.split('\t')
            homo_edge.append([i+1, int(l[2])])
            edge_type.append(l[3])
            pos_seq.append(l[1])
        homo_edges.append(homo_edge)
        edge_types.append(edge_type)
        pos_seqs.append(pos_seq)
    return homo_edges, edge_types, pos_seqs


if __name__ == '__main__':
    argparse
    depparser = CoreNLPDependencyParser(url='http://localhost:9000')

    # processing train set
    file = '../../data/CCAT50/processed/author_0,1_sent_2_train.csv'
    df = pd.read_csv(file)
    homo_edges, hetoro_edges, pos_seqs = get_dep_edges(df['text'])
    df['homo_edges'] = homo_edges
    df['hetoro_edges'] = hetoro_edges
    df['pos_seqs'] = pos_seqs
    df.to_csv(file, index=False)