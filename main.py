import copy
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
import time
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams
from nltk.corpus import brown

from pre_processing import pre_process_data
from utils import pos_tag

def train_hmm(tagged_words, final_tags):
    
    # Returns model parameters of emission probability and transition probability

    # Count distribution of all tags in the corpus
    tokens, tags = zip(*tagged_words)
    tagCounter = Counter(tags)
    
    
    #Count of each token with corresponding tag
    tokenTags = defaultdict(Counter)
    for token, tag in tagged_words:
        tokenTags[token][tag] +=1
    
    # Count of all tag bigrams
    tagTags = defaultdict(Counter)
    bigrams = list(ngrams(tags, 2))

    for tag1, tag2 in bigrams:
        tagTags[tag1][tag2] += 1
    
    # Calculating emission probabilities from the tag count and token tag distribution
    emissionProbs = defaultdict(Counter)

    for token, tag in tagged_words:
        emissionProbs[token][tag] = tokenTags[token][tag] / tagCounter[tag]

    # Calculating transition probabilities using the bigram frequencies
    
    transition_probs = pd.DataFrame(np.zeros((len(final_tags), len(final_tags))), index = final_tags, columns = final_tags)

    for tag1 in transition_probs.index:
        for tag2 in transition_probs.columns:

            try:
                transition_probs.loc[tag1][tag2] = (tagTags[tag1][tag2])/tagCounter[tag1]
            except: 
                transition_probs.loc[tag1][tag2] = 0

    return transition_probs, emissionProbs, tagCounter, tokenTags, tagTags

if __name__ == "__main__":

    corpora = brown.tagged_sents()
    corpora = list(corpora)

    corpora, tags = pre_process_data(corpora)

    # splitting sentences into words for training
    tagged_words = [[word,tag] for sent in corpora for word,tag in sent]

    transition_probs, emissionProbs, tagCounter, tokenTags, tagTags = train_hmm(tagged_words, tags)

    prediction = pos_tag('I love spicy food', transition_probs, emissionProbs, tags)

    print(prediction)
    