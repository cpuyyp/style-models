from collections.abc import Mapping
from typing import List, Optional, Sequence, Union
from tqdm.auto import trange, tqdm
import pandas as pd
import ast
import json
import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.data.data import BaseData
from torch.utils.data.dataloader import default_collate


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'alignments':
            return self.num_nodes
        elif key == 'segment_ids':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['alignments', 'segment_ids']:
            return 0
        if key in ['pos', 'text']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
        

class MyDataset(Dataset):
    # this only works for segment length >= 2.
    # for single sentence, there might be long sentences dropped. then the data becomes empty
    # Currently, this code cannot handle that.
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, max_length=256):
        self.max_length = max_length
        self.length = len(os.listdir(f'{root}/raw'))
        self.raw_files = [f"{i}.json" for i in range(self.length)]
        self.processed_files = [f"{i}.pt" for i in range(self.length)]
        with open('/home/jz17d/Desktop/dependency2id.json') as f:
            self.relation2id = json.load(f)

        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return self.processed_files

    def process(self):
        idx = 0
        error_count = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with open(raw_path) as f:
                curr = json.load(f)
                
            data = MyData()      
            data.y = torch.tensor(curr['author_id'] if 'author_id' in curr else curr['author'])
            data.doc_id = torch.tensor(curr['doc_id'])  

            # text_ids = bert_tokenizer(curr['text'], max_length=512, padding=True, truncation=True, return_tensors='pt')
            # for key in text_ids:
            #     data[f'text_{key}'] = text_ids[key]

            temp_datalist = []
            for j in range(len(curr['edge_indexs'])):
                temp_data = MyData()
                temp_data.edge_index = torch.cat([torch.LongTensor([[0],[0]]),  # for self loop of CLS token
                                             torch.LongTensor(curr['edge_indexs'][j]).T, 
                                             # for batching purpose, if data.x is missing, edge_index is used to inference batch
                                             # an isolated node (the SEP in this case) will mess all up
                                             torch.LongTensor([[len(curr['edge_indexs'][j])+1],[len(curr['edge_indexs'][j])+1]])], 
                                            axis=1)
                temp_data.edge_type_ids = torch.LongTensor([36]+[self.relation2id[t.split(':')[0]] for t in curr['hetoro_edges'][j]]+[36])
                if temp_data.edge_index.shape[1] >= self.max_length-1:
                    error_count += 1
                    continue

                # pos_ids = pos_tokenizer(' '.join(curr['pos_seqs'][j]), max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
                # for key in pos_ids:
                #     temp_data[f'pos_{key}'] = pos_ids[key].squeeze(0)

                # # need to manually create position_ids for pos bert input
                # temp_data.pos_position_ids = torch.arange(temp_data.pos_input_ids.shape[0])
                
                # these two must be strings
                temp_data.text = curr['text'][j]
                temp_data.pos = ' '.join(curr['pos_seqs'][j])

                temp_data.num_syllables = torch.LongTensor([17]+curr['num_syllables'][j]+[17]) # 17 for CLS and SEP
                temp_data.num_chars = torch.LongTensor([0]+curr['num_chars'][j]+[0]) # 0 for CLS and SEP
                temp_data.word_freqs = torch.LongTensor([10]+curr['word_freqs'][j]+[10]) # 10 for CLS and SEP
                
                temp_data.alignments = torch.LongTensor([-1]+curr['alignments'][j]+[curr['alignments'][j][-1]+1]) + 1 # taking care of the CLS and SEP
                temp_data.num_nodes = len(temp_data.edge_type_ids)
                temp_datalist.append(temp_data)
                
            temp_batch = Batch.from_data_list(temp_datalist)

            for key in temp_batch.keys:
                data[key] = temp_batch[key]
            data.num_nodes = len(data.edge_type_ids)
            data.segment_ids = torch.zeros(len(data.text), dtype=torch.long)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'{idx}.pt'))
            idx += 1

        print(f'{error_count} sentences dropped because of exceeding max_length {self.max_length}')
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{idx}.pt'))
        return data

# edit based on pyg source code
class SegmentBatchCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch): 
        
        elem = batch[0]
        if isinstance(elem, BaseData):
            out = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
            for key in ['text', 'pos']:
                out[key] = [elem for sublist in out[key] for elem in sublist]
                # it's not clear for me what are these two dicts used for.
                # the two lines of code are not verified 
                # out._slice_dict[key] = out.segment_ids 
                # out._inc_dict[key] = torch.zeros_like(out.segment_ids, dtype=torch.long)
            return out
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]
            
        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)
    
# edit based on pyg source code
class SegmentDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData]],
        batch_size: int = 4,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=SegmentBatchCollater(follow_batch, exclude_keys),
            **kwargs,
        )


# this is used to set the keys of json loaded dictionary back to integers.
# especially for test_docid2index
def keystoint(x):
    return {int(k): v for k, v in x.items()}

def get_seg_loader(dataset, 
                   segment_length,
                   split,
                   batch_size=32, 
                   shuffle=True, 
                   max_length=256,
                   num_workers=5,
                   pin_memory=True,
                   persistent_workers=True):
    dataset_locations = {
        'imdb62': '/scratch/data_jz17d/data/imdb62/',
        'blogs50': '/scratch/data_jz17d/data/blogs50/',
        'ccat50': '/scratch/data_jz17d/data/ccat50/'
    }
    location = dataset_locations[dataset]

    if isinstance(split, list): # can be a list. e.g. also train with train + val dataset
        datasets = []
        for splt in split:
            dir = f'{location}/segment_{segment_length}_{splt}' if segment_length != 'doc' else f'{location}/doc_{splt}'
            mydataset = MyDataset(dir)
            datasets.append(mydataset)
        loader = SegmentDataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    else:
        dir = f'{location}/segment_{segment_length}_{split}' if segment_length != 'doc' else f'{location}/doc_{split}'
        mydataset = MyDataset(dir)
        loader = SegmentDataLoader(mydataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return loader
