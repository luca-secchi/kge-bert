
import json
import os
import pickle
import numpy as np

import fire
import torch
import logging

from torch.optim import Adam
from sklearn.metrics import classification_report

from config import LEARNING_RATE, MAX_SEQ_LENGTH, VALID_META
from data_utils import get_best_thresholds
from experiment import Experiment
from models import LinearMultiClassifier, ExtraMultiClassifier

import pandas as pd
import random


logging.basicConfig(level=logging.INFO)

### See https://discuss.pytorch.org/t/cant-make-bert-model-training-result-reproducible/102024
def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define experiments

experiments = {
    
    # bert-base-uncased
    'airbnb__bert': Experiment(
        'a', 'bert-base-uncased', with_text=True, with_manual=False, with_1hot_vec=False
    ),
    'airbnb__bert_mask': Experiment(
        'a', 'bert-base-uncased-airbnb_london_20220910-mask', with_text=True, with_manual=False, with_1hot_vec=False
    ),    
    'airbnb__bert_meta': Experiment(
        'a', 'bert-base-uncased', with_text=True, with_manual=True, with_1hot_vec=False
    ),
    'airbnb__bert_meta_mask': Experiment(
        'a', 'bert-base-uncased-airbnb_london_20220910-mask', with_text=True, with_manual=True, with_1hot_vec=False
    ),    
    'airbnb__bert_meta_hot_encoding': Experiment(
        'a', 'bert-base-uncased', with_text=True, with_manual=True, with_1hot_vec=True
    ),
    'airbnb__bert_meta_hot_encoding_mask': Experiment(
        'a', 'bert-base-uncased-airbnb_london_20220910-mask', with_text=True, with_manual=True, with_1hot_vec=True
    ),    
    'airbnb__meta_hot_encoding': Experiment(
        'a', 'bert-base-uncased', with_text=False, with_manual=True, with_1hot_vec=True,
        classifier_model= ExtraMultiClassifier
    ),
    'airbnb__hot_encoding': Experiment(
        'a', 'bert-base-uncased', with_text=False, with_manual=False, with_1hot_vec=True,
        classifier_model= ExtraMultiClassifier
    ),
    'airbnb__bert_hot_encoding': Experiment(
        'a', 'bert-base-uncased', with_text=True, with_manual=False, with_1hot_vec=True
    ),
    'airbnb__bert_hot_encoding_mask': Experiment(
        'a', 'bert-base-uncased-airbnb_london_20220910-mask', with_text=True, with_manual=False, with_1hot_vec=True
    ),

        # baseline
    'airbnb__linear_meta_hot_encoding': Experiment(
        'a', '-', with_text=False, with_manual=True, with_1hot_vec=True,
        classifier_model=LinearMultiClassifier
    ),
    
}

experiments["airbnb__bert_ext"] = experiments["airbnb__bert"]
experiments["airbnb__bert_ext_v2"] = experiments["airbnb__bert"]
experiments["airbnb__linear_all_meta_hot_encoding"] = experiments["airbnb__linear_meta_hot_encoding"]

experiments["airbnb__bert_all_meta_hot_encoding_mask"] = experiments["airbnb__bert_meta_hot_encoding_mask"]
experiments["airbnb__bert_all_meta_hot_encoding"] = experiments["airbnb__bert_meta_hot_encoding"]
experiments["airbnb__bert_all_meta_norm_hot_encoding"] = experiments["airbnb__bert_meta_hot_encoding"]
experiments["airbnb__bert_ext_all_meta_norm_hot_encoding"] = experiments["airbnb__bert_meta_hot_encoding"]
experiments["airbnb__bert_meta_norm_hot_encoding"] = experiments["airbnb__bert_meta_hot_encoding"]
experiments["airbnb__bert_all_meta_mask"] = experiments["airbnb__bert_meta_mask"]
experiments["airbnb__bert_all_meta"] = experiments["airbnb__bert_meta"]
experiments["airbnb__bert_all_meta_norm"] = experiments["airbnb__bert_meta"]
experiments["airbnb__bert_meta_norm"] = experiments["airbnb__bert_meta"]
experiments["airbnb__all_meta_hot_encoding"] = experiments["airbnb__meta_hot_encoding"]
experiments["airbnb__all_meta_norm_hot_encoding"] = experiments["airbnb__meta_hot_encoding"]
experiments["airbnb__meta_norm_hot_encoding"] = experiments["airbnb__meta_hot_encoding"]

experiments["airbnb__bert_valid_meta_norm_hot_encoding"] = experiments["airbnb__bert_meta_hot_encoding"]
experiments["airbnb__bert_valid_meta_norm"] = experiments["airbnb__bert_meta"]
experiments["airbnb__valid_meta_norm_hot_encoding"] = experiments["airbnb__meta_hot_encoding"]

def save_reports(prefix, experiment, outputs, output_ids, t_max, y):
    if len(experiment.labels) == 2:
        labels = [experiment.labels[0]]
        gold_y = y[:,0]
    else:
        labels = experiment.labels
        gold_y = y

    # Save val outputs
    df_val_outputs = pd.DataFrame(outputs, columns=labels)
    df_val_outputs["id"] = output_ids
    #df_val_outputs["threshold"] = t_max
    df_val_outputs["gold"] = gold_y
    df_val_outputs["prediction"] = np.where(outputs>t_max, 1, 0)
    df_val_outputs.to_csv(os.path.join(experiment.get_output_dir(), prefix+'_outputs.csv'), index=False)
    
    prediction = np.where(outputs>t_max, 1, 0)
    # Prepare reports for validation set 
    report = classification_report(gold_y, prediction, target_names=experiment.labels, output_dict=True)
    report_str = classification_report(gold_y, prediction, target_names=experiment.labels)

    # Save dev reports for validation set
    with open(os.path.join(experiment.get_output_dir(), prefix+'_report.json'), 'w') as f:
        json.dump(report, f)
    
    with open(os.path.join(experiment.get_output_dir(), prefix+'_report.txt'), 'w') as f:
        f.write(report_str)


def run_on_val_and_test(name, cuda_device, extras_dir, df_train_path, df_val_path, df_test_path, output_dir, 
                        epochs=None, seq_length=MAX_SEQ_LENGTH, continue_training=False, batch_size=None, 
                        max_training_size=None, max_dev_size=None, max_test_size=None, 
                        lookup_name="", random_state=1, mkdir=False, just_validate=False, existing_model_path=None, 
                        max_repeat=0, evaluate_train = False, dropout=0.1, use_valid_meta = False, 
                        enable_cache= False, embedding_cache_file=None, update_embedding_cache=False):
    

    set_seed(seed=random_state)

    tot_epochs = epochs ## The number of epochs to run if no repetition is performed

    if type(max_training_size) is str:
        print("Train size limit disables")
        max_training_size = None 

    if name not in experiments:
        print(f'Experiment not found: {name}')
        exit(1)

    if use_valid_meta:
        valid_meta = VALID_META
    else:
        valid_meta = None

    print("Max repeat: %s" % max_repeat)
    print("Dropout: %s" % dropout)
    experiment = experiments[name]
    experiment.name = name
    experiment.lookup_name = lookup_name
    experiment.output_dir = output_dir
    experiment.random_state = random_state
    experiment.enable_cache = enable_cache
    if enable_cache:
        if embedding_cache_file is None:
            print("Creating empty cache.")
            experiment.cache_embedding = {}
            experiment.cache_masked_tokens = {}
        else:
            try:
                print("Opening embaddign cache file")
                file_path = os.path.join(extras_dir, embedding_cache_file)
                with open(file_path, 'rb') as f:
                        (cache_embedding, cache_masked_tokens) = pickle.load(f)
                        print("Loaded embedding cache")
                        print("Num keys: ",len(list(cache_embedding.keys())))
                experiment.cache_embedding = cache_embedding
                experiment.cache_masked_tokens = cache_masked_tokens
            #except Exception as e:
            except FileNotFoundError:
                print("Error loading embedding cache. File not found %s. Setting empty one." % file_path)
                experiment.cache_embedding = {}
                experiment.cache_masked_tokens = {}
    else:
        print("!!! Embedding cache disables.")
        experiment.cache_embedding = {}
        experiment.cache_masked_tokens = {}
    
    experiment.init(cuda_device, epochs, batch_size, continue_training, mkdir=mkdir, max_repeat=max_repeat, dropout=dropout)

    train_dataloader, val_dataloader, _, val_df, val_y, train_y = experiment.prepare_data_loaders(df_train_path, df_val_path, extras_dir, max_training_size=max_training_size, max_val_size=max_dev_size, seq_length=seq_length, random_state=random_state, valid_meta=valid_meta)

    model = experiment.get_model()

    print(f'Using model: {type(model).__name__}')

    # Load existing model weights
    if continue_training or just_validate:
        print('Loading existing model weights...')
        if existing_model_path:     
            model.load_state_dict(torch.load(existing_model_path))
        else:
            model.load_state_dict(torch.load(os.path.join(experiment.get_output_dir(), 'model_weights')))

    # Training
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print("Learing rate epsilon: ", LEARNING_RATE)

    # Model to GPU
    model = model.cuda()
    
    if not just_validate: ## We are not just validating but also training
        model, tot_epochs, avg_loss_history = experiment.train(model, optimizer, train_dataloader)

        if update_embedding_cache and embedding_cache_file is not None:
            store_tuple = (model.cache_embedding, model.cache_masked_tokens)
            with open(os.path.join(extras_dir, embedding_cache_file), 'wb') as f:
                pickle.dump(store_tuple, f)

        print("__train_epochs:%s__" % tot_epochs)
        avg_loss_history_df = pd.DataFrame(avg_loss_history, columns=['loss'])
        avg_loss_history_df['epoch'] = range(len(avg_loss_history))
        avg_loss_history_df['delta_loss'] = avg_loss_history_df['loss'].diff()
        avg_loss_history_df.to_csv(os.path.join(experiment.get_output_dir(), 'avg_loss_history.csv'), index=False)

        always_decreasing = (avg_loss_history_df['delta_loss'].fillna(0) <= 0).all()
        ### print avg_loss_history_df 
        print("__avg_loss_always_decreasing:%s__" % always_decreasing)
        print("Avg Loss history:\n",  avg_loss_history_df.to_string(index=False))

        if evaluate_train:
            _, train_eval_dataloader, _, _, train_y, _ = experiment.prepare_data_loaders(df_train_path, df_train_path, extras_dir, max_training_size=max_training_size, max_val_size=max_training_size, seq_length=seq_length, random_state=random_state, valid_meta=valid_meta)

            # Output for train data
            train_output_ids, train_outputs = experiment.eval(model, train_eval_dataloader)
            t_base = [0.5] * len(experiment.labels) #base threshold for each label is
            save_reports('train', experiment, train_outputs, train_output_ids, 0.5, train_y)

    # Validation for hyperparameters tuning (threshold)
    output_ids, outputs = experiment.eval(model, val_dataloader)
    
    t_max, f_max = get_best_thresholds(experiment.labels, val_y, outputs, plot=False)
    print(f'Best threshold: {t_max}, best F1: {f_max}')

    save_reports('val', experiment, outputs, output_ids, t_max, val_y)

    
    ### We prepare the data loader for test set. WARNING we must leave the default test_set=False because with true the gold value vector is set to zero
    _, test_dataloader, vec_found_selector, test_df, test_y, train_y = experiment.prepare_data_loaders(df_train_path, df_test_path, extras_dir, max_training_size=max_training_size, max_val_size = max_test_size, seq_length=seq_length, random_state=random_state, valid_meta=valid_meta)

    # Test for model evaluation
    test_output_ids, test_outputs = experiment.eval(model, test_dataloader)

    save_reports('test', experiment, test_outputs, test_output_ids, t_max, test_y)

    
    with open(os.path.join(experiment.get_output_dir(), 'best_thresholds.csv'), 'w') as f:
        f.write(','.join([str(t) for t in t_max]))

    with open(os.path.join(experiment.get_output_dir(), 'outputs_with_ids.pickle'), 'wb') as f:
        pickle.dump((test_outputs, test_output_ids), f)

    torch.save(model.state_dict(), os.path.join(experiment.get_output_dir(), 'model_weights'))

    with open(os.path.join(experiment.get_output_dir(), 'model_config.json'), 'w') as f:
        json.dump(model.config, f)
    


if __name__ == '__main__':
    fire.Fire()