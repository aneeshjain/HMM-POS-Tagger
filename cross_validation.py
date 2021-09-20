from pre_processing import pre_process_data
from nltk.corpus import brown
import random
import time
from utils import test_set_acc
from main import train_hmm



#Getting the full corpus
corpora1 = brown.tagged_sents()
corpora1 = list(corpora1)
# Randomly sampling 10000 sentences since cross validation over full corpus takes a lot of time
#corpora1 = random.sample(corpora1, 10000)

#Running pre processing on data to remove out of scope tags and add START and END tags
corpora1, tags = pre_process_data(corpora1)

# Creating data folds for cross validation
num_folds = 5
len_fold = len(corpora1)/num_folds
random.shuffle(corpora1)
data_folds = [corpora1[x:x+int(len_fold)] for x in range(0, len(corpora1), int(len_fold))]

# Running cross validation

fold_accuracies = []

total_time = 0

for fold_i in range(len(data_folds)):
    
    t1 = time.time()
    print("Working on Fold {}...".format(fold_i+1))

    test_fold = data_folds[fold_i]
    train_fold = [item for fold in range(len(data_folds)) if fold != fold_i for item in data_folds[fold]]
    
    # splitting sentences into words
    tagged_words = [[word,tag] for sent in train_fold for word,tag in sent]
    
    transition_probs, emissionProbs, tagCounter, tokenTags, tagTags = train_hmm(tagged_words, final_tags)

    fold_acc = test_set_acc(test_fold, transition_probs, emissionProbs, tags)
    
    print("Accuracy for Fold {}: {}".format(fold_i+1, fold_acc))
    
    fold_accuracies.append(fold_acc)
    
    t2 = time.time()
    
    delta = (t2-t1)/60
    
    
    total_time+=delta
    
    print("Time taken for Fold {}: {} minutes".format(fold_i+1, delta))
    print('\n')


print("All fold accuracies: ", fold_accuracies)
print("\n")
print("Total Time Taken: {} minutes".format(total_time))


cross_val_acc = sum(fold_accuracies)/len(fold_accuracies)
print("5-Fold Cross Validation Accuracy: ", cross_val_acc)