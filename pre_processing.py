import numpy as np

#Preprocessing data
def pre_process_data(corpora):
    
    #Adding START and END tag to every sentence
    for sent in corpora:
        sent.insert(0, ('START', 'START'))
        sent.insert(len(sent), ('END', 'END'))

    unique_tags = {tag for sent in corpora for word,tag in sent}

    #tags we want to include
    ques_tags =  ['NI', 'RP', 'OD', 'TO', 'NR', 'WP', 'PP', 'DT', 'QL', 
    'AB', 'HV', 'CD', 'IN', 'JJ', 'AP', 'BE', 'PN', 'AT', 'WR', 'RN', 
    'EX', 'VB', 'MD', 'WD', 'START', 'WQ', 'NP', 'UH', 'NN', 'CS', 'RB', 
    'FW', 'END', 'DO', 'CC', ',', 'PPSS', 'VBD', 'WDT']
    ques_tags = np.array(ques_tags)

    # Mask for only those tags that we want to include
    mask = [True if x in unique_tags else False for x in ques_tags]
    mask = np.array(mask)

    final_tags = ques_tags[mask]
    
    # Replacing all other tags that we dont want to include
    for i in range(len(corpora)):
        for j in range(len(corpora[i])):
            corpora[i][j] = list(corpora[i][j])
            if corpora[i][j][1] not in final_tags:
                corpora[i][j][1] = 'UNK'

    final_tags = np.append(final_tags,'UNK')

    return corpora, final_tags