# KGE-BERT for Accomodation Offers Optimization

Implementation of the paper *Optimizing Tourism Accommodation Offers by Integrating Language Models and Knowledge Graph Technologies* (under review).

In case of problems, feel free to contact us or submit a GitHub issue.

## Content

- CLI script to run all experiments
- a script to extract DBpedia entities from accommodations' description
- a notebook to process AirBnB files with TAO to produce training datasets and one hot encoding vectors files

## Model architecture

![KGE-BERT](images/model-architecture.png)


## Installation

Requirements:
- Python 3.8
- CUDA GPU

Install dependencies:
```
pip install -r requirements.txt
```

Download pre trained BERT model under *pre-trained_models* as explained in the relative [README.md file](pre-trained_models/README.md)

## Reproduce paper results

Download London listings data from InsideAirBnb (we used a dump for date 2022-09-10) and put it in data/london/date=20220910/ directory.

Process the csv using async_dbpedia_spotlight.py script with an instruction like:
```
python async_dbpedia_spotlight_copy.py data/london/date=20220910/listings.csv.gz data/london/date=20220910/london_listings_mapped.csv --text_column description --types tourism
```

Create one-hot encoding vectors and the train, dev and test set using KG-BERT_data_engineering.ipynb notebook. The notebook saves the necessary files under bert_input_data.


Train and evaluate the model with all experiments' settings by running the following scripts:
```
## Experiments: KGE-BERT-full, KGE-BERT-1hot, KGE-BERT-num
bash run_all_KGE-BERT_experiments_no_text_injection.sh 

## Experiments: KGE-BERT-injected-full
bash run_KGE-BERT-injected-full.sh

## Experiment: BERT, Logistic Regression
bash run_BERT_and_baseline_experiments.sh

## Experiment: BERT-injected
bash run_BERT-injected.sh
```

The results would be saved under *output_results*.

## Train the model for other tourist destinations

You can simply follow the previous instructions to train the model for a different tourist destination. You just need to downolad a different listing file from Inside Airbnb.
Consider that the current code is based on english text anaysis. It could be adapted to other languages using a different BERT base model that supports multilingual and using the appropriate language option when performing entity linking using Dbpedia spotlight (script async_dbpedia_spotlight). Change the script following the instruction in https://github.com/MartinoMensio/spacy-dbpedia-spotlight

## How to cite

If you are using the code in this repository, please cite [our paper]():
```

```

## References

The code in this repository is based on [pytorch-bert-document-classification](https://github.com/malteos/pytorch-bert-document-classification)

## License

MIT


