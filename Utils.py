# -*- coding: utf-8 -*-
"""
#### DATA FORMATIING AND RESULTS PRINTING #####
@author: Khaled Boughanm
"""

import numpy as np

"""
Routine to format the data in the appropriate design
"""
def Format_data(rawdocs):
    # the documents as bags of words
    docs  = map(lambda x: x.split(" "), rawdocs)
    
    # unique words
    vocab = list(set([item for sublist in docs for item in sublist]))
    
    # replace words in documents with wordIDs
    docs = [[vocab.index(y) for y in x] for x in docs]

    # data array formatting 
    data_size = np.sum([len(doc) for doc in docs])
    data = np.zeros((int(data_size), 3), dtype=np.int)
    i = 0
    for doc_idx, doc in enumerate(docs):
        word_pos = 0
        for word_idx, word in enumerate(doc):
            data[i][0] = int(doc_idx)
            data[i][1] = int(word_pos)
            data[i][2] = int(word)
            i += 1
            word_pos += 1
    
    
    max_len_doc = max([len(doc) for doc in docs])
    V = len(vocab) # vocabulary size
    num_docs = len(docs)
    return({'data': data,
            'max_len_doc' : max_len_doc,
            'V' : V,
            'num_docs': num_docs,
            'vocab': vocab})
            


"""
 Routines for results description
"""
def describe_topics(n, phi, vocab):
    '''
    n:     the number of words to print to describe the topic
    phi:   the matrix of topic word distribution
    vocab: the vocabulary used
    '''
    topics = [sorted([(vocab[k], phi_topic[k]) for k in range(len(vocab))], key=lambda tup: tup[1],reverse =True)[:n] for phi_topic in phi]
    return(topics)
    
def describe_docs(n, lim, theta, phi, vocab):
    '''
    n:     the number of words to print to describe the topic
    lim:   number of words top of the topics of each document
    phi:   the matrix of topic word distribution
    vocab: the vocabulary used
    '''
    topics = describe_topics(n, phi, vocab)
    res = {}
    for d, doc in enumerate(theta):
        doc_topics = sorted([ (k, doc[k]) for k in range(len(topics))], key=lambda tup: tup[1], reverse = True)[:lim]
        res[d] = [([w[0] for w in topics[topic[0]][:5]],round(topic[1],2)) for topic in doc_topics]
        
    return(res)