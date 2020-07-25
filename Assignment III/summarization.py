#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import nltk
from itertools import combinations
nltk.download('punkt')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


# In[2]:


# Reading file and dropping unrelevant columns
print("Reading mapping file...",end="")
mapping = pd.read_csv("04-10-mag-mapping.csv", low_memory=False)
mapping.drop(mapping.columns.difference(['cord_uid','abstract']), 1, inplace=True)
print("OK!")


# In[3]:


# Converting data to a dictionary 
mapping.set_index("cord_uid", inplace=True)


# In[4]:


# Popping cord_id that does not have abstracts
abstracts = mapping.to_dict()["abstract"]

for cord_id, abstract in list(abstracts.items()):
    if type(abstract) == type(float('nan')):
        abstracts.pop(cord_id, None)
    else:
        abstracts[cord_id] = abstract.lower()


# In[18]:


# Creating Document Frequency
df = {}
# A dictionary to hold list of tokens of respective cord_id
doc_tokens = {}
i = 0
print("Creating document frequencies and idf among all documents:")
# Iterating throough the dictionary to get document frequencies
for cord_id, abstract in abstracts.items():
    if i%10000 == 0: print(i, "/", len(abstracts),end="\r")
    tokens = nltk.word_tokenize(abstract)
    doc_tokens[cord_id] = tokens
    tokens = set(tokens)
    for token in tokens:
        if token in df:
            df[token] += 1
        else:
            df[token] = 1
    i += 1
print(len(abstracts),"/",len(abstracts),end=" ...")
# Converting document frequencies to inver document frequencies
idf = {}
N = len(abstracts)
for k,v in df.items():
    idf[k] = np.log10(N / v)
print("OK!")


# In[7]:


print("Reading relations file...",end="")
# Reading relations file
relations = open("qrels-rnd1.txt", 'r').read()
print("OK!")


# In[8]:


relations = relations.split("\n")
relations.pop(-1)


# In[9]:


# Getting relevant cord_id s that are relevant to selected topics
relevant_1 = []
relevant_13 = []
relevant_23 = []
for rel in relations:
    spl = rel.split()
    if spl[0] == '1' and spl[3] == '2':
        relevant_1.append(spl[2])
    elif spl[0] == '13' and spl[3] == '2':
        relevant_13.append(spl[2])
    elif spl[0] == '23' and spl[3] == '2':
        relevant_23.append(spl[2])


# In[10]:


# Creating vocab list
vocab = list(idf.keys())
vocab.sort()


# In[19]:


def log10_plus1(n):
    """
    Returns respective tf score for given frequency.
    0 -> 0, 1->1 ... so on
    """
    if n == 0:
        return 0
    else:
        return np.log10(n) + 1


# In[20]:


def cos_sim(vec1, vec2):
    """
    Returns the cosine similarity between given two vectors
    """
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# In[21]:


def power_method(trans_p):
    """
    Given a transition probability matrix this method calculates the x vector with power method.
    The final x is returned
    """
    x = np.ones((1,trans_p.shape[0])) / trans_p.shape[0]
    count = 0
    while True:
        newx = np.matmul(x,trans_p)
        if np.sum(np.abs(newx - x)) < 0.00001:
            break
        count += 1
        x = newx
    print(" Converged after {} steps.".format(count))
    return x


# In[47]:


def sentences_of_top10_doc(relevant):
    """
    Given a relevant list of cord_id s this method returns the sentences of top 10 PageRanked documents as a
    list, top 10 documents' ids and their PageRank scores.
    """
    print("\tCreating tfidf vector:")
    # Creating tfidf vector
    tfidf = {}
    i = 0
    for doc_id in relevant:
        if i % int(len(relevant) / 10) == 0: print("\t" + str(i),"/",len(relevant),end="\r")
        tfidf[doc_id] = []
        try:
            for term in vocab:
                # Calculating tfidf score by multiplying idf and tfidf
                tfidf[doc_id].append(log10_plus1(doc_tokens[doc_id].count(term)) * idf[term])
        except KeyError:
            # The docs that don't have abstracts in csv file are removed from the list
            # as they raise KeyError.
            tfidf.pop(doc_id)
            i += 1
            continue
        i += 1
        
    print("\t" + str(len(relevant)),"/",len(relevant),end=' ...')
    print("OK!")
    # The sorted list of cord_id s to get a staticly placed elements
    docs = list(tfidf.keys())
    docs.sort()
    
    print("\tCalculating similarities and creating transition probability matrix...",end="")
    # Creating transition matrix. If the cosine similalrity between 2 docs is greater than 0.1 then
    # the transition between two docs is set to True
    transition = np.zeros((len(docs),len(docs)),dtype=np.int8)
    # As the edges are always two way or in other words the graph is undirected the reflection of the
    # cord_id pairs with respect to main diagonal
    # Hence calculating cosine similarity of the combinations of two docs is enough 
    for rowid, colid in combinations(docs,2):
        tmp_sim = cos_sim(np.array(tfidf[rowid]), np.array(tfidf[colid]))
        if tmp_sim > 0.1:
            i = docs.index(rowid)
            j = docs.index(colid)
            transition[i][j] = 1
            transition[j][i] = 1

    # Calculating transition probability matrix with teleportation rate
    tp_rate = 0.15
    trans_p = np.zeros(transition.shape)
    for i, row in enumerate(transition):
        if np.sum(row) == 0:
            trans_p[i] = 1 / row.shape[0]
        else:
            trans_p[i] = tp_rate * (1 / row.shape[0]) + (row / np.sum(row)) * (1 - tp_rate)
    print("OK!")
    
    print("\tPower Method:",end="")
    # Getting final x vector of power method
    x = power_method(trans_p)
    print()
    # Getting top 10 id s with respect to PageRank score
    top10 = x[0].argsort()[-10:][::-1]
    
    # All sentences of top 10 documents is concatanated and split into sentences
    sentences = " ".join([abstracts[docs[i]] for i in top10])
    sentences = sent_detector.tokenize(sentences)
    return sentences, [docs[i] for i in top10], x[0][top10]


# In[48]:


def top20_sentences(sentences):
    """
    Given a list of sentences, this method calculates the PageRank scores of them
    and returns the top 20 sentences among them and and their PageRank scores.
    """
    
    print("\tCalculating idf of sentences...",end="")
    # Document frequency for sentences is calculated
    sdf = {}
    sent_tokens = {}
    for i, sent in enumerate(sentences):
        tokens = nltk.word_tokenize(sent)
        sent_tokens[i] = tokens
        tokens = set(tokens)
        for token in tokens:
            if token in sdf:
                sdf[token] += 1
            else:
                sdf[token] = 1
                
    # Converting into idf 
    sidf = {}
    sN = len(sentences)
    for k,v in sdf.items():
        sidf[k] = np.log10(sN / v)
    print("OK!")

    # Vocab of sentences is sorted
    svocab = list(sidf.keys())
    svocab.sort()
    
    print("\tCreating tfidf scores for sentences...",end="")
    # Tfidf score for sentences is calculated
    stfidf = {}
    for i, sent in enumerate(sentences):
        stfidf[i] = []
        for term in svocab:
            stfidf[i].append(log10_plus1(sent_tokens[i].count(term)) * sidf[term])
    print("OK!")
    print("\tCalculating cosine similarities of sentences and creating transition probability matrix...",end="")
    # Transition matrix for sentences are calculated as sam above
    stransition = np.zeros((len(sentences),len(sentences)),dtype=np.int8)
    for rowid, colid in combinations([i for i in range(len(sentences))],2):
        tmp_sim = cos_sim(np.array(stfidf[rowid]), np.array(stfidf[colid]))
        if tmp_sim > 0.1:
            stransition[rowid][colid] = 1
            stransition[colid][rowid] = 1

    # Teleportation has taken into consideration
    strans_p = np.zeros(stransition.shape)
    tp_rate = 0.15
    for i, row in enumerate(stransition):
        if np.sum(row) == 0:
            strans_p[i] = 1 / row.shape[0]
        else:
            strans_p[i] = tp_rate * (1 / row.shape[0]) + (row / np.sum(row)) * (1 - tp_rate)
    print("OK!")
    print("\tPower Method:",end="")
    # Power method is applied among sentences
    sx = power_method(strans_p)

    # Top 20 sentence ids is returned
    top20sentences = sx[0].argsort()[-20:][::-1]
    return top20sentences, sx[0][top20sentences]


# In[49]:


print("Summarizing for Topic 1:")
sentences1, top10ids1, top10scores1 = sentences_of_top10_doc(relevant_1)
sids1, top20scores1 = top20_sentences(sentences1)


# In[52]:


with open("Topic1Results.txt", 'w') as f:
    f.writelines("Top 10 documents and their scores:\n")
    for i in range(len(top10ids1)):
        f.writelines(top10ids1[i] + ",\t" + str(top10scores1[i]) + "\n")
    f.writelines("\nTop 20 sentences and their scores:\n")
    for j, i in enumerate(sids1):
        f.writelines(sentences1[i] + ",\t" + str(top20scores1[j]) + "\n")


# In[49]:


print("\nSummarizing for Topic 13:")
sentences13, top10ids13, top10scores13 = sentences_of_top10_doc(relevant_13)
sids13, top20scores13 = top20_sentences(sentences13)


# In[52]:


with open("Topic13Results.txt", 'w') as f:
    f.writelines("Top 10 documents and their scores:\n")
    for i in range(len(top10ids13)):
        f.writelines(top10ids13[i] + ",\t" + str(top10scores13[i]) + "\n")
    f.writelines("\nTop 20 sentences and their scores:\n")
    for j, i in enumerate(sids13):
        f.writelines(sentences13[i] + ",\t" + str(top20scores13[j]) + "\n")


# In[49]:


print("\nSummarizing for Topic 23:")
sentences23, top10ids23, top10scores23 = sentences_of_top10_doc(relevant_23)
sids23, top20scores23 = top20_sentences(sentences23)


# In[52]:


with open("Topic23Results.txt", 'w') as f:
    f.writelines("Top 10 documents and their scores:\n")
    for i in range(len(top10ids23)):
        f.writelines(top10ids23[i] + ",\t" + str(top10scores23[i]) + "\n")
    f.writelines("\nTop 20 sentences and their scores:\n")
    for j, i in enumerate(sids23):
        f.writelines(sentences23[i] + ",\t" + str(top20scores23[j]) + "\n")


