#!/usr/bin/env python
# coding: utf-8
import os
import math
import random

print("Reading training files...", end=' ', flush=True)
# Create list of training files
train_pos_files = ['train/pos/' + f for f in os.listdir('train/pos')]
train_neg_files = ['train/neg/' + f for f in os.listdir('train/neg')]

# Calculate prior probabilities
pos_prior = math.log(len(train_pos_files) / (len(train_neg_files) + len(train_pos_files)))
neg_prior = math.log(len(train_neg_files) / (len(train_neg_files) + len(train_pos_files)))

def preprocess(train):
    """
    This method replaces punctuations and newline characters with space character
    given a string. Then it splits all words into a list and returns it.
    
    Parameters:
    train -- The string that will be preprocessed
    
    Returns:
    train -- A list containing words
    """
    
    train = train.replace('\n', ' ')
    
    punctuations = [",", ".", ":", "\"", "'", "/", "\\", "*", "=", "-", "_", ")", "(", "[", "]",
               "{", "}", "%", "+", "!", "@", "#", "$", "^", "&", "+", "|", ";", "<",
               ">", "?", "`", "~", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    for punctuation in punctuations:
        train = train.replace(punctuation, " ")
        
    train = train.split()
    
    return train

# Reads training files and adds it iteratively
# Strings that will hold all positive and negative reviews for multinomial naive bayes
train_pos = ""
train_neg = ""
# Lists that will hold all words for binary naive bayes
bin_train_pos = []
bin_train_neg = []
for f in train_pos_files:
    with open(f,'r') as file:
        newfile = file.read()
    # Newly read file is added to according string for multinomial naive bayes
    train_pos = train_pos + newfile
    # Newly read file is preprocessed and converted into set to remove the words that
    # occur more than once. Then it is converted into list again and added to according
    # list for binary naive bayes. Since bernoulli naive bayes needs documents in this way
    # it is also used for bernoulli naive bayes
    bin_train_pos = bin_train_pos + [*set(preprocess(newfile)), ]
        
# Same operations for negative movie reviews
for f in train_neg_files:
    with open(f,'r') as file:
        newfile = file.read()
    train_neg = train_neg + newfile
    bin_train_neg = bin_train_neg + [*set(preprocess(newfile)), ]
    
print("OK!")

# Strings preprocessed and converted into lists for multinomial naive bayes 
train_pos = preprocess(train_pos)
train_neg = preprocess(train_neg)

# Lengts are calculated
train_pos_T = len(train_pos)
train_neg_T = len(train_neg)

bin_train_pos_T = len(bin_train_pos)
bin_train_neg_T = len(bin_train_neg)

# Vocab is created
vocab = set(train_pos + train_neg)
T = len(vocab)

def count_map(train):
    """
    Given a list of words, this method calculates a count map of that list in the following form
    key=word : value=count_of_word
    
    Parameters:
    train -- A list of words
    
    Returns:
    count -- A map that have words as keys and count of that words as values
    """
    count = {}
    for word in train:
        if word in count:
            count[word] += 1
        else:
            count[word] = 1
            
    for word in vocab:
        if word not in count:
            count[word] = 0
    return count

# Creates count maps of lists
pos_count = count_map(train_pos)
neg_count = count_map(train_neg)

bin_pos_count = count_map(bin_train_pos)
bin_neg_count = count_map(bin_train_neg)

print("Calculating likelihoods of words...", end=' ', flush=True)
# For every word in vocabulary likelihood dictionary for all naive bayes models is created
pos_likelihood = {}
neg_likelihood = {}
bin_pos_likelihood = {}
bin_neg_likelihood = {}
bern_pos_likelihood = {}
bern_neg_likelihood = {}
# Iterate through vocabulary
for w in vocab:
    # Likelihood for multinomial naive bayes is calulated
    pos_likelihood[w] = math.log((pos_count[w] + 1) / (train_pos_T + T))
    # Likelihood for binary naive bayes is calulated
    bin_pos_likelihood[w] = math.log((bin_pos_count[w] + 1) / (bin_train_pos_T + T))
    # Likelihood for bernoulli naive bayes is calculated
    bern_pos_likelihood[w] = (bin_pos_count[w] + 1) / (len(train_pos_files) + 2)
    
    # Same calculations for negative reviews
    neg_likelihood[w] = math.log((neg_count[w] + 1) / (train_neg_T + T))
    bin_neg_likelihood[w] = math.log((bin_neg_count[w] + 1) / (bin_train_neg_T + T))
    bern_neg_likelihood[w] = (bin_neg_count[w] + 1) / (len(train_neg_files) + 2)
print("OK!")

print("Reading test files...", end=' ', flush=True)
# Creates list of training file
test_pos_files = ['test/pos/' + f for f in os.listdir('test/pos')]
test_neg_files = ['test/neg/' + f for f in os.listdir('test/neg')]
print("OK!")

def predict(doc):
    """
    Given a string this method calculates probabilites of being in positive and negative classes
    using multinomial likelihoods.
    It returns 1 if it is more likely to be positive and 0 otherwise
    
    Parameters:
    doc -- A string that its probability of being positive or negative calculated
    
    Returns:
    1 -- If it is more likely to be positive
    0 -- Otherwise
    """
    doc = preprocess(doc)
    pos_prob = pos_prior
    neg_prob = neg_prior
    for w in doc:
        if w in vocab:
            pos_prob += pos_likelihood[w]
            neg_prob += neg_likelihood[w]

    if pos_prob >= neg_prob:
        return 1
    else:
        return 0

def bin_predict(doc):
    """
    Given a string this method calculates probabilites of being in positive and negative classes
    using binary likelihoods.
    It returns 1 if it is more likely to be positive and 0 otherwise
    
    Parameters:
    doc -- A string that its probability of being positive or negative calculated
    
    Returns:
    1 -- If it is more likely to be positive
    0 -- Otherwise
    """
    doc = [*set(preprocess(doc)), ]
    pos_prob = pos_prior
    neg_prob = neg_prior
    for w in doc:
        if w in vocab:
            pos_prob += bin_pos_likelihood[w]
            neg_prob += bin_neg_likelihood[w]

    if pos_prob >= neg_prob:
        return 1
    else:
        return 0

def bern_predict(doc):
    """
    Given a string this method calculates probabilites of being in positive and negative classes
    using bernoulli likelihoods.
    It returns 1 if it is more likely to be positive and 0 otherwise
    
    Parameters:
    doc -- A string that its probability of being positive or negative calculated
    
    Returns:
    1 -- If it is more likely to be positive
    0 -- Otherwise
    """
    doc = [*set(preprocess(doc)), ]
    pos_prob = pos_prior
    neg_prob = neg_prior
    for w in vocab:
        if w in doc:
            pos_prob += math.log(bern_pos_likelihood[w])
            neg_prob += math.log(bern_neg_likelihood[w])
        else:
            pos_prob += math.log(1 - bern_pos_likelihood[w])
            neg_prob += math.log(1 - bern_neg_likelihood[w])
    if pos_prob >= neg_prob:
        return 1
    else:
        return 0

print("Predicting test files with 3 naive bayes models concurrently(2 minutes approximately)...", end=' ', flush=True)
# Variables are created for confusion matrices for all 3 models
tp = 0
tn = 0
fp = 0
fn = 0
bin_tp = 0
bin_tn = 0
bin_fp = 0
bin_fn = 0
bern_tp = 0
bern_tn = 0
bern_fp = 0
bern_fn = 0

# Lists that will hold outputs of the models
output = []
bin_output = []
bern_output = []

# Iterating through test files and predicting for all 3 models and fill variables accordingly
for f in test_pos_files:
    with open(f,'r') as file:
        test = file.read()
    
    result = predict(test)
    if result == 1:
        tp += 1
    else:
        fn += 1
    output.append(result)
    
    result = bin_predict(test)
    if result == 1:
        bin_tp += 1
    else:
        bin_fn += 1
    bin_output.append(result)
    
    result = bern_predict(test)
    if result == 1:
        bern_tp += 1
    else:
        bern_fn += 1
    bern_output.append(result)

for f in test_neg_files:
    with open(f,'r') as file:
        test = file.read()
        
    result = predict(test)
    if result == 0:
        tn += 1
    else:
        fp += 1
    output.append(result)
    
    result = bin_predict(test)
    if result == 0:
        bin_tn += 1
    else:
        bin_fp += 1
    bin_output.append(result)
    
    result = bern_predict(test)
    if result == 0:
        bern_tn += 1
    else:
        bern_fp += 1
    bern_output.append(result)
print("OK!")

print("Performance scores are being calculated...", end=' ', flush=True)
# Performance and scores are calculated for multinomial naove bayes
pos_precision = tp / (tp + fp)
pos_recall = tp / (tp + fn)
pos_f = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall)

neg_precision = tn / (tn + fn)
neg_recall = tn / (tn + fp)
neg_f = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall)

macro_precision = (pos_precision + neg_precision) / 2
micro_precision = (tn + tp) / (tn + tp + fn + fp)

macro_recall = (pos_recall + neg_recall) / 2
micro_recall = micro_precision

macro_f = (pos_f + neg_f) / 2
micro_f = micro_precision

# Performance and scores are calculated for binary naove bayes
bin_pos_precision = bin_tp / (bin_tp + bin_fp)
bin_pos_recall = bin_tp / (bin_tp + bin_fn)
bin_pos_f = (2 * bin_pos_precision * bin_pos_recall) / (bin_pos_precision + bin_pos_recall)

bin_neg_precision = bin_tn / (bin_tn + bin_fn)
bin_neg_recall = bin_tn / (bin_tn + bin_fp)
bin_neg_f = (2 * bin_neg_precision * bin_neg_recall) / (bin_neg_precision + bin_neg_recall)

bin_macro_precision = (bin_pos_precision + bin_neg_precision) / 2
bin_micro_precision = (bin_tn + bin_tp) / (bin_tn + bin_tp + bin_fn + bin_fp)

bin_macro_recall = (bin_pos_recall + bin_neg_recall) / 2
bin_micro_recall = bin_micro_precision

bin_macro_f = (bin_pos_f + bin_neg_f) / 2
bin_micro_f = bin_micro_precision

# Performance and scores are calculated for bernoulli naove bayes
bern_pos_precision = bern_tp / (bern_tp + bern_fp)
bern_pos_recall = bern_tp / (bern_tp + bern_fn)
bern_pos_f = (2 * bern_pos_precision * bern_pos_recall) / (bern_pos_precision + bern_pos_recall)

bern_neg_precision = bern_tn / (bern_tn + bern_fn)
bern_neg_recall = bern_tn / (bern_tn + bern_fp)
bern_neg_f = (2 * bern_neg_precision * bern_neg_recall) / (bern_neg_precision + bern_neg_recall)

bern_macro_precision = (bern_pos_precision + bern_neg_precision) / 2
bern_micro_precision = (bern_tn + bern_tp) / (bern_tn + bern_tp + bern_fn + bern_fp)

bern_macro_recall = (bern_pos_recall + bern_neg_recall) / 2
bern_micro_recall = bern_micro_precision

bern_macro_f = (bern_pos_f + bern_neg_f) / 2
bern_micro_f = bern_micro_precision

print("OK!")

def swap_element(list1, list2, index):
    """
    Swaps element between list1 and list2 at given index
    
    Parameters:
    list1 -- First list
    list2 -- Second list
    index -- Index the swap operation will be applied
    
    Returns:
    list1 -- First list with swapped element
    list2 -- Second list with swapped element
    """
    temp = list1[index]
    list1[index] = list2[index]
    list2[index] = temp
    return list1, list2

def compute_s_star(output1, output2):
    """
    Computes difference of micro-averaged F1 score for given outputs.
    Ground truth is a list, first 300 element is positive and last 300 element is negative.
    
    Parameters:
    output1 -- Output of first model
    output2 -- Output of second model
    
    Returns:
    Difference of micro-averaged F1 score
    """
    count1 = 0
    count2 = 0
    for i in range(300):
        if output1[i] == 1:
            count1 += 1
        if output2[i] == 1:
            count2 += 1
    for i in range(300, 600):
        if output1[i] == 0:
            count1 += 1
        if output2[i] == 0:
            count2 += 1
    return abs((count1 / 600) - (count2 / 600))
    
def randomization_test(output1, output2, s):
    """
    Computes approximate randomization test for given two outputs.
    
    Parameters:
    output1 -- Output of first model
    output2 -- Output of second model
    s -- Difference of micro-averaged F1 scores of given two outputs
    
    Returns:
    p -- p value of randomization test
    """
    count = 0
    R = 1000
    for i in range(R):
        temp1 = output1.copy()
        temp2 = output2.copy()
        for j in range(len(temp1)):
            if random.random() < 0.5:
                temp1, temp2 = swap_element(temp1, temp2, j)
        if compute_s_star(temp1, temp2) >= s:
            count += 1
    p = (count + 1) / (R + 1)
    return p

print("Randomization tests are being done on outputs of 3 models...", end=" ", flush=True)
# p values are calculated
mult_bin_p = randomization_test(output, bin_output, abs(micro_f - bin_micro_f))
bin_bern_p = randomization_test(bin_output, bern_output, abs(bin_micro_f - bern_micro_f))
mult_bern_p = randomization_test(output, bern_output, abs(micro_f - bern_micro_f))
print("OK!")

print("Multinomial Naive Bayes Results:")
print("\t Macro-Averaged:")
print("\t\tPrecision:", macro_precision)
print("\t\tRecall:", macro_recall)
print("\t\tF1 Score:", macro_f)
print("\t Micro-Averaged:")
print("\t\tPrecision:", micro_precision)
print("\t\tRecall:", micro_recall)
print("\t\tF1 Score:", micro_f)

print("Binary Naive Bayes Results:")
print("\t Macro-Averaged:")
print("\t\tPrecision:", bin_macro_precision)
print("\t\tRecall:", bin_macro_recall)
print("\t\tF1 Score:", bin_macro_f)
print("\t Micro-Averaged:")
print("\t\tPrecision:", bin_micro_precision)
print("\t\tRecall:", bin_micro_recall)
print("\t\tF1 Score:", bin_micro_f)

print("Bernoulli Naive Bayes Results:")
print("\t Macro-Averaged:")
print("\t\tPrecision:", bern_macro_precision)
print("\t\tRecall:", bern_macro_recall)
print("\t\tF1 Score:", bern_macro_f)
print("\t Micro-Averaged:")
print("\t\tPrecision:", bern_micro_precision)
print("\t\tRecall:", bern_micro_recall)
print("\t\tF1 Score:", bern_micro_f)

print("\n\n")
print("Approximate Randomization Tests:")
print("\tMultinomial and Binary", mult_bin_p)
print("\tMultinomial and Bernoulli", mult_bern_p)
print("\tBinary and Bernoulli", bin_bern_p)
