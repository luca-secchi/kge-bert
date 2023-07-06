import os
import pickle

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch import nn
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from data_utils import get_num_and_1hot_vec, to_dataloader, TensorIndexDataset
from config import MAX_SEQ_LENGTH, HIDDEN_DIM, MLP_DIM, TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCHS, \
    default_extra_cols, BERT_MODELS_DIR
from models import ExtraBertMultiClassifier, BertMultiClassifier
import inspect

class Experiment(object):
    """
    Holds all experiment information
    """
    name = None
    output_dir = None
    epochs = None
    batch_size = None
    device = None
    labels = None

    def __init__(self, task, bert_model, classifier_model=None, with_text=True,
                 with_manual=True, with_1hot_vec=True, author_vec_switch=False, mlp_dim=None):
        self.task = task
        self.bert_model = bert_model
        self.with_text = with_text
        self.with_manual = with_manual
        self.with_1hot_vec = with_1hot_vec
        self.author_vec_switch = author_vec_switch
        self.classifier_model = classifier_model
        
        self.mlp_dim = mlp_dim if mlp_dim is not None else MLP_DIM
        self.extra_cols = None

    def init(self, cuda_device, epochs, batch_size, continue_training, mkdir=True, max_repeat=0, dropout=0.1):
        print("--------- Experiment init ---------")

        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        if not torch.cuda.is_available():
            print('CUDA GPU is not available')
            exit(1)

        self.epochs = epochs if epochs is not None else NUM_TRAIN_EPOCHS
        self.batch_size = batch_size if batch_size is not None else TRAIN_BATCH_SIZE
        self.max_repeat = max_repeat # max number of times an epoch can be repeated if the avg_train_loss does not decrease 
        print("Max repeat: %d" % self.max_repeat)
        self.dropout = dropout
        print("Dropout: %f" % self.dropout)

        output_dir = os.path.join(self.get_output_dir())
        if mkdir: ## if we have to create the directory check if it already exists
            if not continue_training and os.path.exists(output_dir):
                print(f'Output directory exist already: {output_dir}')
                exit(1)
            else:
                os.makedirs(output_dir)
        ## if we use an existing directory we check if a model weight file is already present and abort if we are not continuing the trainign using it
        else: 
            ### Exit if the outpu dir does not exist
            if not os.path.exists(output_dir):
                print("We should not create an output directory and the output directory configured does not exist: %s" % output_dir)
                exit(1)
            
            print("We are using an existing directory: %s" % output_dir )
            weight_file = os.path.join(output_dir, 'model_weights')
            if not continue_training and os.path.isfile(weight_file):
                print("You are not continuing training but there is a weight file in the output directory: %s" % weight_file)
                exit(1)

    def get_output_dir(self):
        return os.path.join(self.output_dir, self.name)

    def get_bert_model_path(self):
        return os.path.join(BERT_MODELS_DIR, self.bert_model)

    def get_1hot_dim(self):
        # Use author switch?
        if self.author_vec_switch:
            vec_1hot_dim = self.vec_1hot_dim + 1
        else:
            vec_1hot_dim = self.vec_1hot_dim

        return vec_1hot_dim

    def get_extra_cols(self):
        if self.with_manual:
            if self.extra_cols is None:
                print("Warning: extra cols is not set, using default columns.")
                extra_cols = default_extra_cols
            else:
                extra_cols = self.extra_cols
        else:
            print("Not using numeric features vector.")
            extra_cols = []
        return extra_cols

    def prepare_data_loaders(self, df_train_path, df_val_path, extras_dir, test_set=False, max_training_size=None, max_val_size=None, seq_length=MAX_SEQ_LENGTH, random_state=1, valid_meta = None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.with_text:
            tokenizer = BertTokenizer.from_pretrained(self.get_bert_model_path(), do_lower_case=False)
        else:
            tokenizer = None

        # Load external data
        if self.with_1hot_vec:
            print("########## Opening vector lookup file:",os.path.join(extras_dir, self.lookup_name))
            with open(os.path.join(extras_dir, self.lookup_name), 'rb') as f:
                item2vec = pickle.load(f)
            self.vec_1hot_dim = len(next(iter(item2vec.values()))) ## all embedding vector have the same size
            print(f'Embeddings avaiable for {len(item2vec)} authors')
        else:
            item2vec = None



        # Load training data
        with open(df_train_path, 'rb') as f:
            train_df, extra_cols, task_b_labels, task_a_labels = pickle.load(f)
            ### rename old names for data colums
            train_df.rename(columns={"authors":"item_ids"}, inplace=True)
            if type(valid_meta) is list: ## we have a list of valid metadata, lets filter the extra_cols
                print("Filtering metadata using valid metadata list")
                extra_cols = list(set(extra_cols).intersection(set(valid_meta)))
                extra_cols.sort()

            if max_training_size is not None:
                train_df = train_df[:max_training_size] ## we take first max_training_size samples 
                #print(train_df[0:10])
                train_df = train_df.sample(n=max_training_size, random_state=random_state) ## shuffle     
        if type(extra_cols) in (list, np.ndarray):
            print("setting extra cols to:", extra_cols)
            
            self.extra_cols = extra_cols
        else:
            print("WARNING extra cols is not a list: ", extra_cols)
            print("extra_cols type: ",type(extra_cols))
            
        # Define labels (depends on task)
        if self.task == 'a':
            self.labels = task_a_labels
        elif self.task == 'b':
            self.labels = task_b_labels
        else:
            raise ValueError('Invalid task specified')

        if self.with_manual or self.with_1hot_vec:
            train_extras, vec_found_count, _ = get_num_and_1hot_vec(
                train_df,
                self.get_extra_cols(),
                item2vec,
                with_vec=self.with_1hot_vec,
                on_off_switch=self.author_vec_switch
            )
        else:
            train_extras = None

        if self.with_text:
            train_texts = [t + '.\n' + train_df['text'].values[i] for i, t in enumerate(train_df['title'].values)]
            print(f"Train dataset contains {len(train_texts)} texts.")

        else:
            train_texts = None


        train_y = train_df[self.labels].values
        
        train_dataloader = to_dataloader(train_texts, train_extras, train_y,
                                         tokenizer,
                                         seq_length,
                                         self.batch_size,
                                         dataset_cls=TensorDataset,
                                         sampler_cls=RandomSampler)

        # Load validation data
        with open(df_val_path, 'rb') as f:
            val_df, _, _, _ = pickle.load(f)
            ### rename old names for data colums
            val_df.rename(columns={"authors":"item_ids"}, inplace=True)
            if max_val_size is not None:
                val_df = val_df[:max_val_size] ## we take first max_training_size samples 
                val_df = val_df.sample(n=max_val_size, random_state=random_state) ## shuffle    
        

        if self.with_manual or self.with_1hot_vec:
            val_extras, vec_found_count, vec_found_selector = get_num_and_1hot_vec(
                val_df,
                self.get_extra_cols(),
                item2vec,
                with_vec=self.with_1hot_vec,
                on_off_switch=self.author_vec_switch,
            )
        else:
            val_extras = None
            vec_found_selector = None

        if self.with_text:
            val_texts = [t + '.\n' + val_df['text'].values[i] for i, t in enumerate(val_df['title'].values)]
        else:
            val_texts = None

        # Is test set?
        # np.zeros((len(test_texts), len(labels)))
        if test_set:
            val_y = np.zeros((len(val_texts), len(self.labels)))
        else:
            val_y = val_df[self.labels].values

        val_dataloader = to_dataloader(val_texts, val_extras, val_y,
                                       tokenizer,
                                       seq_length,
                                       self.batch_size,
                                       dataset_cls=TensorIndexDataset,
                                       sampler_cls=SequentialSampler)

        return train_dataloader, val_dataloader, vec_found_selector, val_df, val_y, train_y

    def get_model(self):
        
        extras_dim = len(self.get_extra_cols())
        
        if self.with_1hot_vec:
            extras_dim += self.get_1hot_dim()

        if self.classifier_model is None:
            # No pre-defined model

            if extras_dim > 0:
                model = ExtraBertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count=len(self.labels),
                    hidden_dim=HIDDEN_DIM,
                    extras_dim=extras_dim,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout,
                    enable_cache=self.enable_cache,
                    cache_embedding=self.cache_embedding,
                    cache_masked_tokens=self.cache_masked_tokens,
                    random_state=self.random_state
                )
            else:
                # Text only: Standard BERT classifier
                model = BertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count=len(self.labels),
                    hidden_dim=HIDDEN_DIM,
                    dropout=self.dropout,
                    enable_cache=self.enable_cache,
                    cache_embedding=self.cache_embedding,
                    cache_masked_tokens=self.cache_masked_tokens,
                    random_state=self.random_state
                )
        elif inspect.isclass(self.classifier_model):
            model = self.classifier_model(labels_count=len(self.labels), extras_dim=extras_dim)
        else:
            model = self.classifier_model

        return model

    def train(self, model, optimizer, train_dataloader):
        prev_avg_loss = 100 ## set hig value so that the first avg_loss would be smaller
        max_repeat = self.max_repeat
        print("Max repeat:", max_repeat)
        total_epocs = 0
        avg_loss_history = []
        for epoch_num in range(self.epochs):
            model.train()
            print(f'Epoch: {epoch_num + 1}/{self.epochs}')

            # for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            repeat = 0
            do_repeat = True
            while do_repeat:
                total_epocs = total_epocs + 1
                train_loss = 0
                if repeat >0:
                        print(f'repeating epoch {epoch_num + 1}' )


                for step_num, batch_data in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    # print("len batch:", len(batch_data))
                    if self.with_text and (
                            self.with_manual or self.with_1hot_vec):
                        # Full features
                        # for t in batch_data:
                        #     print("type:", type(t))
                        token_ids, masks, extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                        # print("token_ids", token_ids.shape)
                        probas = model(token_ids, masks, extras)
                    elif self.with_text:
                        # Text only
                        token_ids, masks, gold_labels = tuple(t.to(self.device) for t in batch_data)
                        probas = model(token_ids, masks)
                    else:
                        # Extras only
                        extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                        probas = model(extras)
                # print("gold labels dim:", gold_labels.shape)
                    if gold_labels.shape[1] == 2: ## in case of binary classification we force sigmoid and single element output
                        gold_labels = gold_labels[:,0].unsqueeze(1) ## just take the first label because the second must be complementary to 1
                    loss_func = nn.BCELoss()
                    batch_loss = loss_func(probas, gold_labels)
                    train_loss += batch_loss.item()

                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    # clear_output(wait=True)
                repeat = repeat + 1
                avg_train_loss = train_loss / len(train_dataloader)
                avg_loss_history.append(avg_train_loss)

                print(f'\r{epoch_num} average loss: {avg_train_loss}')
                #print(f'\r{epoch_num} loss: {train_loss / (step_num + 1)}')

                print(str(torch.cuda.memory_allocated(self.device) / 1000000) + 'M')

                if avg_train_loss < prev_avg_loss:
                    do_repeat = False
                    prev_avg_loss = avg_train_loss 
                
                if repeat >= max_repeat:
                    do_repeat = False
                    print("Max repeat reached")
        
        print(f'Training completed for {total_epocs} epocs')
        
        return model, total_epocs, avg_loss_history

    def eval(self, model, data_loader):

        # Validation
        model.eval()

        output_ids = []
        outputs = None

        with torch.no_grad():
            for step_num, batch_item in enumerate(data_loader):
                batch_ids, batch_data = batch_item

                if self.with_text and (
                        self.with_manual or self.with_1hot_vec):
                    # Full features
                    token_ids, masks, extras, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks, extras)
                elif self.with_text:
                    # Text only
                    token_ids, masks, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks)
                else:
                    # Extras only
                    extras, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(extras)

                numpy_logits = logits.cpu().detach().numpy()

                if outputs is None:
                    outputs = numpy_logits
                else:
                    outputs = np.vstack((outputs, numpy_logits))

                output_ids += batch_ids.tolist()

        print(f'Evaluation completed for {len(outputs)} items')

        return output_ids, outputs
