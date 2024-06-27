from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from numpy import sort

from datetime import date
import spacy
from spacy.matcher import PhraseMatcher

import re
from owlready2 import *

import numpy as np                                
import pandas as pd      
from time import gmtime, strftime  
import json
import os
from urllib.parse import urlparse
from collections import OrderedDict, Counter

def pre_process(text): 
    ### FROM: https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf 
    # lowercase
    text=text.lower() 
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)   
    
    # Swaps line breaks for spaces, also remove apostrophes & single quotes
    text.replace("\n", " ").replace("'"," ").replace("’"," ")
    return text

def produce_balanced_dataset(df, rev_threshold, col='review_scores_value'):
    top = df['top'] = (df[col] > rev_threshold)
    g = pd.DataFrame({'index':top.index, 'value':top.values}).groupby('value')
    res = g.apply(lambda x: x.sample(g.size().min()))['index']
    selected_elements = list(pd.concat([top[list(res[False])],top[list(res[True])]]).index)
    balanced_data = df.loc[selected_elements].sample(frac=1)
    return balanced_data.copy(), top

def produce_balanced_dataset_safe(df, rev_threshold, safe_zone, col, sample_seed = 1):
    ## remove rows with reviews scores value near the threshold
    safe_filter = (df[col]< rev_threshold - safe_zone ) | (df[col]> rev_threshold + safe_zone)
    unsafe_filter = ~safe_filter
    safe_df = df[safe_filter].copy()
    unsafe_df = df[unsafe_filter].copy()
    top = safe_df['__y__'] = (safe_df[col] > rev_threshold)
    unsafe_df['__y__'] = (unsafe_df[col] > rev_threshold)
    g = pd.DataFrame({'index':top.index, 'value':top.values}).groupby('value')
    res = g.apply(lambda x: x.sample(g.size().min(), random_state=sample_seed))['index']
    selected_elements = list(pd.concat([top[list(res[False])],top[list(res[True])]]).index)
    balanced_data = safe_df.loc[selected_elements].sample(frac=1, random_state=sample_seed)
    return balanced_data.copy(), balanced_data["__y__"], unsafe_df

def clean_name(base):
    ## AirBnb amenities name can have a comment after the real name
    ## e.g. "keypad - check yourself into the home with a door code"
    ## e.g. "clothing storage: closet"
    ## we take just the first part
    
    ## we remove the "'s" strings also
    if type(base) != str:
        return None
    base_left = base.split(" - ")[0].split(" – ")[0].split(": ")[0].replace("’s","")
    
    
    result = re.sub("['/,!@#$>()]", ' ', base_left)
    return " ".join(result.split()).lower().strip() ## remove multiple spaces

class AmenityMapper:
    def __init__(self, amenity_classes: pd.DataFrame, class_colummn = 'amenity_class', class_label = 'label'):
        self.__class_colummn = class_colummn
        self.__nlp = spacy.load("en_core_web_sm")
        self.__matcher = PhraseMatcher(self.__nlp.vocab)
        amenity_classes["amenity_id"] = amenity_classes[class_label].apply(lambda e: self.__clean_name(str(e)).lower().strip())
        amenity_classes.drop_duplicates(subset=['amenity_id'])
        self.amenity_classes = amenity_classes
        
        known_amenities = list(amenity_classes['amenity_id'])
        self.known_amenities = known_amenities
        self.patterns = [self.__nlp(am) for am in known_amenities]
        self.__matcher.add("AMENITY_PATTERN", self.patterns)
   
    def __clean_name(self, base):
        return clean_name(base)
    
    def amenity_linker(self, amenity_description: str):
        if type(amenity_description) is not str:
            return None, None
        doc = self.__nlp(amenity_description)
        matches = self.__matcher(doc)
        matched_label = None
        max_len = 0
        for match_id, start, end in matches:
            span = doc[start:end]
            span_len = len(span.text)
            if span_len > max_len:
                matched_label = span.text
                max_len = span_len
                #print(f"Found entity: {span.text}")
        if matched_label is None:
            return None, None
        matched_class = self.amenity_classes.loc[self.amenity_classes["amenity_id"] == matched_label][self.__class_colummn]
        return matched_label, list(matched_class)[0]
    
    def amenity_linker_multi(self, text: str, simple = False):
        if type(text) is not str:
            return []
        doc = self.__nlp(text)
        matches = self.__matcher(doc)
        entities = []
        max_len = 0
        non_overlapping_spans = []
        spans = [doc[start:end] for _, start, end in matches]
        for span in spacy.util.filter_spans(spans):
            #print(span.start, span.end, span.text)
            non_overlapping_spans.append(span)
        
        
        for span in non_overlapping_spans:
            matched_label = span.text
            matched_classes = self.amenity_classes.loc[self.amenity_classes["amenity_id"] == matched_label][self.__class_colummn]
            matched_class = list(matched_classes)[0]
            #entity = {"class": matched_class, "sf": matched_label, "span": span }
            entity = {'surface_char_pos': [span.start_char, span.end_char], 'surface_form': span.text, 'surface_word_pos': [span.start, span.end], 'types': [], 'url': matched_class}
            if simple:
                entities.append(entity['class'])
            else:
                entities.append(entity)
        
        return entities
    
def remap_amenities(am_descriptions, am_dict, output='label'):
    entities = []
    if output == 'label':
        col = 0
    elif output == 'class':
        col = 1
    else:
        raise Exception("Output type unknown: %s" % output)
    for desc in am_descriptions:
        clean_desc = clean_name(desc).lower().strip()
        ent = am_dict[clean_desc][col]
        if ent is not None:
            entities.append(ent)
    return entities

def clean_price(price):
    n_p = price.copy()
    n_p = n_p.str.replace("$","", regex=False)
    n_p = n_p.str.replace(",","", regex=False).astype(np.float64)
    return n_p

def converter(entities, filter = False, prefix=False, full=False):
    if type(entities) is str:
        try:
            list_entities = eval(entities)
        except (SyntaxError, NameError):
            return []
    if type(entities) is list:
        list_entities = entities
    
    if len(list_entities) == 1 and type(list_entities[0]) == str:
        list_entities = []
    if full:
        return list_entities
    
    uris = []
    try:
        for e in list_entities:
            url = urlparse(e["url"])
            keep = True
            if type(filter) is str:
                keep = False
                for entity_type in  e["types"]:
                    if filter in entity_type.lower():
                        keep = True
                        break
            if keep:
                if prefix:
                    uris.append(e["url"])
                else: 
                    if url.fragment != '': ## URI schema using fragments "#rdf_resource_name'
                        uris.append(url.fragment)
                    else: ## URI schema uses last path '/rdf_resource_name'
                        uris.append(os.path.basename(url.path))
    except:
        print(list_entities, type(list_entities))
    return uris

def property_frequencies(df_column, min_freq = 100):
    properties_freq_df = df_column.explode().value_counts().to_frame(name="freq").reset_index(names=["property"]).sort_values(by=['freq'], ascending=False)
    frequent_properties = set(properties_freq_df[properties_freq_df['freq'] > min_freq ]['property'])
    return frequent_properties

def calc_days_since(my_date, last_date, col=None):
    try:
        conv_date = my_date.date()
        res = (last_date - conv_date).days
        return res
    except TypeError as e:
        #print("Column %s Wrong date %s of type %s" % (col,my_date, type(my_date)))
        #return pd.NaT
        return -1

def deb(am_list):
    ### handle Type error when converting lists to unique element lists
    try:
        return list(set(am_list))
    except TypeError:
        #print(am_list, type(am_list))
        return []
    
def root_sublclasses_tree(root_class):
    root_map = {}
    for c in root_class.subclasses():
        root_name = c.name
        for descendant in c.descendants():
            try:
                if root_name not in root_map[descendant.name][0]:
                    root_map[descendant.name][0].append(root_name)
            except KeyError:
                root_map[descendant.name] = [[root_name]]
            
    
    return root_map
            

def get_all_labels(owl_class, languages = ["en"], altLabels = False):
    ##TODO: generalize to extract labels from a list of annotation properties passed as an argument
    candidate_labels = []
    labels = []
    rdfs_labels = list(owl_class.label)
    if altLabels:
        try:
            skos_altLabels = list(owl_class.altLabel) 
        except AttributeError:
            print("######## skos!!!")
            skos_altLabels = []
        skos_annotated_labels = get_annotated_labels(owl_class, skos.altLabel)
    else:
        skos_annotated_labels = []
        skos_altLabels = []
    candidate_labels = rdfs_labels + skos_altLabels + skos_annotated_labels
    #print("examining labels")
    #print(candidate_labels)
    for label in candidate_labels:
        try:
            lang = label.lang
            #print("lang:", lang)
            if lang not in languages:
                #print("###########")
                continue
        except Exception as e:
            print(e)
            #pass
        if label.lower() not in labels and label != owl_class.name: ## avboid duplicated labels
            labels.append(label.lower())
        #else:
            #print("already known", label)
            
    return labels

def accuracy_results_old(X, y, features = "numeric", random_state = 42, model_copy = None, scoring = {'acc': 'accuracy'}):
    if model_copy is None:
        model_copy = model
        
    results = []
    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)
    X_train = pd.DataFrame(normalize(X_train_unnorm, norm='l2'))
    X_test = pd.DataFrame(normalize(X_test_unnorm, norm='l2'))

    for do_norm in [True, False]:
        if not do_norm:
            X_train = X_train_unnorm
            X_test = X_test_unnorm
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
        scores = cross_validate(estimator=model_copy, X=X_train, y=y_train,  
                             scoring=scoring, cv=kfold, n_jobs=-1, return_train_score=True)

        my_scores = { "features": features, "normalisation": do_norm }
        for k, v in scores.items():
            my_scores[k+"_cv_values"] = v
            my_scores[k+"_mean"] = v.mean()
            my_scores[k+"_std"] = v.std()
        results.append(my_scores)
 
    return results


### new ####
def accuracy_results(X, y, model, features = "numeric", random_state = 42, scoring = {'acc': 'accuracy'}, disable_norm = False):
    
    model_copy = model
        
    results = []
    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)
    X_train = pd.DataFrame(normalize(X_train_unnorm, norm='l2'))
    X_test = pd.DataFrame(normalize(X_test_unnorm, norm='l2'))
    #scoring_metrics = {"accuracy": "accuracy", "f1": "f1"}

    for do_norm in [True, False]:
        if do_norm and disable_norm:
            continue
        if not do_norm:
            X_train = X_train_unnorm
            X_test = X_test_unnorm
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
        scores = cross_validate(estimator=model_copy, X=X_train, y=y_train,  
                             scoring=scoring, cv=kfold, n_jobs=-1, return_train_score=True)
        # if do_norm:
        #     print("Results with normalization")
        # else:
        #     print("Results without normalization")
        
        # print(f"CV scores: {scores}")
        my_scores = { "features": features, "normalisation": do_norm }
        for k, v in scores.items():
            my_scores[k+"_cv_values"] = v
            my_scores[k+"_mean"] = v.mean()
            my_scores[k+"_std"] = v.std()
        results.append(my_scores)
            
        #print(f"CV scores mean: {scores.mean():.4f}")
        #print(f"CV scores std: +/- {scores.std():.4f}")
        #results.append({ "features": features, "normalisation": do_norm, "cv_scores": scores, "accuracy_mean": scores.mean(), "accuracy_std": scores.std()})
    return results

def accuracy_by_feature_reduction(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    # fit model on all training data
    full_model = sklearn.base.clone(model)
    full_model.fit(X_train, y_train)
    # make predictions for test data and evaluate
    y_pred = full_model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    #print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Fit model using each importance as a threshold
    thresholds = sort(full_model.feature_importances_)[::-1][:-1] ### reverse order and skip the last threshold to avoid zzero features selection
    accuracies = []
    num_features = []
    partial_models = []
    for thresh in thresholds:
        select_X_train = X_train.filter(X_train.columns[full_model.feature_importances_ <  thresh], axis="columns")
        select_X_test = X_test.filter(X_test.columns[full_model.feature_importances_ <  thresh], axis="columns")
        if select_X_train.shape[1] == 0:
            accuracies.append(0)
            num_features.append(0)
            continue
        selection_model = sklearn.base.clone(model)
        selection_model.fit(select_X_train, y_train)
        
        partial_models.append(selection_model)
        # eval model
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        num_features.append(select_X_train.shape[1])
        #print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    
    res = {
        "full_model_accuracy": accuracy,
        "full_model": full_model,
        "partial_models": partial_models,
        "thresholds": thresholds,
        "partial_accuracies": accuracies,
        "partial_num_features": num_features
    }
    return res

def extract_res(target, results, results_full):
    rows = []
    for i, r in enumerate(results):
        for k, v in r.items():
            if ("mean" in k) or ("std" in k):
                rows.append({"target": target, "metric": k, "value_all_feaures": results_full[i][k], "value": v, "reduction":results_full[i][k] - v}) 
                #print(k, results_full[i][k], "reduced to ", v , "delta: %.2f" % (results_full[i][k] - v) )
    return rows