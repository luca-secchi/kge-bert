import re

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

try:
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras_preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_fscore_support

import matplotlib.pyplot as plt
import re

import numpy as np
import torch

class TensorIndexDataset(TensorDataset):
    def __getitem__(self, index):
        """
        Returns in addition to the actual data item also its index (useful when assign a prediction to a item)
        """
        return index, super().__getitem__(index)
    
def text_to_train_tensors(texts, tokenizer, max_seq_length):
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:max_seq_length - 1], texts))
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=max_seq_length, truncating="post", padding="post",
                                     dtype="int")

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]

    # to tensors
    # train_tokens_tensor, train_masks_tensor
    return torch.tensor(train_tokens_ids), torch.tensor(train_masks)


def to_dataloader(texts, extras, ys,
                 tokenizer,
                 max_seq_length,
                 batch_size,
                 dataset_cls=TensorDataset,
                 sampler_cls=RandomSampler):
    """
    Convert raw input into PyTorch dataloader
    """
    #train_y = train_df[labels].values

    # Labels
    train_y_tensor = torch.tensor(ys).float()

    if texts is not None and extras is not None:
        # All features
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_extras_tensor = torch.tensor(extras, dtype=torch.float)

        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_extras_tensor, train_y_tensor)
    elif texts is not None and extras is None:
        # Text only
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    elif texts is None and extras is not None:

        train_extras_tensor = torch.tensor(extras, dtype=torch.float)
        train_dataset = dataset_cls(train_extras_tensor, train_y_tensor)
    else:
        raise ValueError('Either texts or extra must be set.')

    train_sampler = sampler_cls(train_dataset)
    
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


def get_num_and_1hot_vec(df, extra_cols, item2vec, with_vec=True, on_off_switch=False):
    """
    Build matrix for extra data 
    """
    if with_vec:
        vec_1hot_dim = len(next(iter(item2vec.values())))

        if on_off_switch:
            vec_1hot_dim += 1  # One additional dimension of binary (1/0) if embedding is available
    else:
        vec_1hot_dim = 0
        
    
    base = df[extra_cols].to_numpy(copy=True) ## transform the metadata columns of the train data frame in numpy array
    extras = np.pad(base, ((0, 0), (0, vec_1hot_dim ))) ## add enough columns on the right to accomodate one-hot encoding vector columns if needed

   
    vec_found_selector = [False] * len(df)
   
    vec_found_count = 0 

    print("########## vec_1hot_dim", vec_1hot_dim)
    print("########## extra_cols len", len(extra_cols))
    print("########## extras shape", extras.shape)

    for i, item_ids in enumerate(df['item_ids']):
        # one-hot encoding vec
        if with_vec:
            for item_id in item_ids.split(';'):
                #print("Look for ",item_id)
                if item_id in item2vec:
                    if on_off_switch:
                        extras[i][len(extra_cols):len(extra_cols) + vec_1hot_dim - 1] = item2vec[item_id]
                        extras[i][len(extra_cols) + vec_1hot_dim] = 1
                    else:
                        extras[i][len(extra_cols):len(extra_cols)+vec_1hot_dim] = item2vec[item_id]

                    vec_found_count += 1
                    vec_found_selector[i] = True
                    break
        
    print("######### vec_found_count", vec_found_count)

    return extras, vec_found_count, vec_found_selector


def get_best_thresholds(labels, test_y, outputs, plot=False):
    """
    Hyper parameter search for best classification threshold
    """
    if len(labels) == 2:
        labels = [labels[0]] ## if we have just 2 labels we have a binary classification, just consider the first one

    t_max = [0] * len(labels)
    f_max = [0] * len(labels)

    for i, label in enumerate(labels):
        ts = []
        fs = []

        for t in np.linspace(0.1, 0.99, num=50):
            p, r, f, _ = precision_recall_fscore_support(test_y[:,i], np.where(outputs[:,i]>t, 1, 0), average='micro')
            ts.append(t)
            fs.append(f)
            if f > f_max[i]:
                f_max[i] = f
                t_max[i] = t

        if plot:
            print(f'LABEL: {label}')
            print(f'f_max: {f_max[i]}')
            print(f't_max: {t_max[i]}')

            plt.scatter(ts, fs)
            plt.show()
            
    return t_max, f_max
