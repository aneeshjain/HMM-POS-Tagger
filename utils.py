import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

def predict_hmm(test_sent, transition_probs, emissionProbs, final_tags):
    
    # Initialising empty delta table for Viterbi Algorithm
    delta_table = np.zeros((len(final_tags), len(test_sent)))
    delta_table

    backpointer = pd.DataFrame(index = final_tags, columns = range(0,len(test_sent)), dtype=object)

    delta_table = pd.DataFrame(delta_table, index = final_tags, columns = [x[0] for x in test_sent])

    # Selcting emmission probabilities for current test sentence
    emission_probs = np.zeros((len(final_tags), len(test_sent)))
    emission_probs = pd.DataFrame(emission_probs, index = final_tags, columns = [x[0] for x in test_sent])
    
    for tag in emission_probs.index:
        for word in range(len(test_sent)):
            
            emission_probs.loc[tag][test_sent[word][0]] = emissionProbs[test_sent[word][0]][tag]
    
    # Initial probability
    
    _pi = pd.Series(np.zeros((len(final_tags))), index = final_tags)
    _pi['START'] = 1
    
   
    # Viterbi Algorithm
    delta_table.loc[:][test_sent[0][0]] = _pi*emission_probs[test_sent[0][0]]
    backpointer.loc[:][0] = '0'

    for t in range(1, len(test_sent)):
            prev_deltas = delta_table.iloc[:, t-1]
            
            delta_table[test_sent[t][0]] = (transition_probs.mul(prev_deltas, axis=0)).max(axis = 0)*emission_probs.iloc[:, t]
            # Updating backpointer
            backpointer[t] = (transition_probs.mul(prev_deltas, axis=0)).idxmax()
    

    answer = []
    
    # Choosing most probable final state to initiate back trace from
    best_path_pointer = delta_table.iloc[:, len(test_sent)-1].idxmax()
    answer = [best_path_pointer]

    # Backtrakcing most probable states
    for i in range(len(test_sent)-1, -1, -1):
        answer.append(backpointer.loc[best_path_pointer][i])
        best_path_pointer = backpointer.loc[best_path_pointer][i]
    answer.reverse()
    
    # Returns answer
    return answer[1:]

def sentence_acc(predicted_pos, test_sent):
    # Returns accuracy for a predicted sentence
    test_pos = [x[1] for x in test_sent]
    
    result = np.array(predicted_pos) == np.array(test_pos)
    
    acc = np.sum(result)/len(predicted_pos)
    
    return acc

def test_set_acc(test_set, transition_probs, emissionProbs, tags):
    
    #Returns accuracy for full test set
    
    total_acc = 0
    
    for sent in test_set:
        predicted_pos = predict_hmm(sent, transition_probs, emissionProbs, tags)
        
        total_acc += sentence_acc(predicted_pos, sent)
    
    return total_acc/len(test_set)


def pos_tag(sent, transition_probs, emissionProbs, tags):
    x = word_tokenize(sent)
    y = ['UNK']*len(x)
    
    z = zip(x,y)
    z = list(z)
    z.insert(0, ('START', 'START'))
    z.insert(len(z), ('END', 'END'))
    
    w = x.copy()
    w.insert(0, 'START')
    w.insert(len(w), 'END')
    
    return list(zip(w, predict_hmm(z, transition_probs, emissionProbs, tags)))

