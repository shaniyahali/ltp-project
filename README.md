# Sentiment Analysis Project for LTP- Comparison of traditional embedding method vs BERT

As part of the project, we have taken an existing traditional embedding model (in this case - **GloVe**). The changes required for BERT is incorporated unto this, while keeping the model unchanged.

## SentimentAnalysis_Using_Glove

### Description
This is used as the base model. It is an implementation of Sentiment Analysis using PyTorch that is taken as is from [Upgraded Sentiment Analysis](
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb) by [bentrevett](https://github.com/bentrevett). The only changes to the original version is to retrieve more metrics from the predictions such as f1-score, confusion matrix and training loss graph

### Running On Google Collab:

It can be run on Google Collab directly from the link provided in the [notebook](https://github.com/shaniyahali/ltp-project/blob/master/SentimentAnalysis_Using_Glove.ipynb).

### Running on local machine:

Disclaimer: We have not run a local version so the below is only a reference for getting it started. Dependencies on cuda version of the pytorch should be taken care of. It is recommended to run the Google Collab notebook as these packages are pre-installed and less prone to error.

To install base packages:
```
pip install numpy
pip install matplotlib
pip install scikit-learn
```

To install PyTorch, see installation instructions on the [PyTorch website](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/pytorch.org).

To install TorchText:
```
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions here making sure to install the English models with:
```
python -m spacy download en
```

## SentimentAnalysis_Using_Bert


### Description
The base model is modified to use BERT instead of Glove word embeddings. We used `pytorch-nlp` to load the IMDB dataset instead of `torchtext` as we need to run specific operations on the sentences to make it in the BERT format. We also used an extremely powerful and user-friendly pre-trained bert adapter - [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) from [hugging-face](https://github.com/huggingface).

### Running On Google Collab:

It can be run on Google Collab directly from the link provided in the [notebook](https://github.com/shaniyahali/ltp-project/blob/master/SentimentAnalysis_Using_Bert.ipynb).

### Running on local machine:

Disclaimer: We have not run a local version so the below is only a reference for getting it started. Dependencies on cuda version of the pytorch should be taken care of. It is recommended to run the Google Collab notebook as these packages are pre-installed and less prone to error.

To install base packages:
```
pip install numpy
pip install matplotlib
pip install scikit-learn
```

To install keras-gpu, see installation instructions on [keras-tensorflow-windows-installation](https://github.com/antoniosehk/keras-tensorflow-windows-installation)

To install PyTorch, see installation instructions on the [PyTorch website](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/pytorch.org).

To install the hugging-face `pytorch-pretrained-bert` adapter and `pytorch-nlp` for the IMDB dataset
```
pip install pytorch_pretrained_bert
pip install pytorch-nlp
```

## Report

Our experiment analysis, results and conclusion can be found in the Report

## Contributors

Shaniya Hassan Ali

Nachiket Desai

Inge Salomons


