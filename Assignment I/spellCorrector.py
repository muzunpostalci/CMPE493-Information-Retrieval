#!/usr/bin/env python
# coding: utf-8

import sys

# Taking input file name from command line
inputfile = sys.argv[-1]

# Reads errors from file.
with open("spell-errors.txt", 'r') as f:
    errors = f.read()
errors = errors.lower()
# Split every line
errors = errors.split('\n')
# Remove empty strings from the list for for loop convenience
while "" in errors:
    errors.remove("")

try:
    with open("test-words-correct.txt", 'r') as f:
        correct = f.read()
    correct = correct.split('\n')
    correct.remove("")
except:
    print("There is not correct test word to read. Disabling results...")


if not inputfile.endswith('.py'):
    print("Reading", inputfile, "...")
    with open(inputfile,'r') as f:
        wrong = f.read()
    wrong = wrong.split('\n')
    wrong.remove("")
    print(inputfile, "is read.\n")
else:
    print("No input file provided or it couldn't be found. Looking for test-words-misspelled.txt...")
    try:
        with open("test-words-misspelled.txt",'r') as f:
            wrong = f.read()
        wrong = wrong.split('\n')
        wrong.remove("")
        print("test-words-misspelled.txt is read.\n")
    except:
        print("test-words-misspelled.txt couldn't be found. Exiting program since there is no misspelled words.")
        sys.exit()


print("corpus.txt is being read.\n")
try:
    with open("corpus.txt", 'r') as f:
        corpus = f.read()
except:
    print("corpus.txt couldn't be found. Exiting program.")
    sys.exit()

# Replace new lines with spaces
corpus = corpus.replace('\n',' ')

print("Processing Corpus...")

punctuations = [",", ".", ":", "\"", "'", "/", "\\", "*", "=", "-", "_", ")", "(", "[", "]",
               "{", "}", "%", "+", "!", "@", "#", "$", "^", "&", "+", "|", ";", "<",
               ">", "?", "`", "~", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Delete all punctuations and numbers from the corpus
for punctuation in punctuations:
    corpus = corpus.replace(punctuation, " ")
    
# Case folding
corpus = corpus.lower()

# Tokenize corpus
corpus = corpus.split()
T = len(corpus)
print("Corpus Tokenized.\n")


# Create dictionary
print('Creating dictionaries...')
dictionary = {}
for word in corpus:
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1

# Create probability P(w) of dictionary
probs = {}
for word, count in dictionary.items():
    probs[word] = count/T
print('Dictionaries Created.\n')


# A simple method to print summary of comparison output with the correct data
def results(correct, output):
    tota = 0
    notin = 0
    couldve = 0
    for i in range(len(output)):
        if output[i] == correct[i]:
            tota+=1
        else:
            if not correct[i] in dictionary:
                notin+=1
            else:
                if DLdistance(correct[i], wrong[i], return_ops=False) == 1:
                    couldve +=1
                    print(correct[i], wrong[i], DLdistance(correct[i], wrong[i], return_ops=False),'\t', output[i])
                
    print("WITH MAX EDIT DISTANCE 1")
    print("\nTotal: 384")
    print("Correct:", tota)
    print("Wrong:", 384-tota)
    print("\tNot in dict:", notin)
    print("\tIn dict:", 384-tota-notin)
    print("\t\tDistance == 1:", couldve)
    print("\t\tDistance > 1:", 384-tota-notin-couldve)


# return_ops returns operations of edits of wordsif set to True
# verbose prints the table for debugging purposes if set to True
def DLdistance(correct, wrong, return_ops=True, verbose=False):
    # Creates empty table for calculating edit distance
    d = [[0 for i in range(len(wrong)+1)] for j in range(len(correct)+1)]
    
    # Filing in default values
    for i in range(len(correct)+1):
        d[i][0] = i
    for j in range(len(wrong)+1):
        d[0][j] = j
    
    # Starts to fill the table
    for i in range(1, len(correct)+1):
        for j in range(1, len(wrong)+1):
            # A list to choose minimum value from
            minlist = [d[i-1][j]+1,
                       d[i][j-1]+1,
                       d[i-1][j-1] if correct[i-1]==wrong[j-1] else d[i-1][j-1]+1]
            d[i][j] = min(minlist)
            
            # Check condition for transposition operation
            if i > 1 and j > 1 and correct[i-1] == wrong[j-2] and correct[i-2]==wrong[j-1]:
                transchecklist = [d[i][j], d[i-2][j-2] + 1]
                d[i][j] = min(transchecklist)
                
    # The last element of the table is the distance
    distance = d[len(correct)][len(wrong)]
    
    # Printing Table
    if verbose:
        print("      ",end='')
        for i in range(len(d[0]) - 1 ):
            print(wrong[i] + "  ", end='')
        print()
        for i in range(len(d)):
            print((" " + correct)[i], d[i])
    
    # Backtracking is done to return operations list
    if return_ops:
        INF = 9999
        operations = []    
        x = len(d) - 1
        y = len(d[0]) - 1 
        while not (x == 0 and y == 0):
            # First operation type is determined
            minlist = []
            if x>0:
                minlist.append(d[x-1][y]+1)
            else:
                minlist.append(INF)
            if y>0:
                minlist.append(d[x][y-1]+1)
            else:
                minlist.append(INF)
            if x>0 and y>0:
                minlist.append(d[x-1][y-1] if correct[x-1]==wrong[y-1] else d[x-1][y-1]+1)
            else:
                minlist.append(INF)
            minindex = minlist.index(min(minlist))
            if minindex == 0:
                operation = 'del'
            elif minindex == 1:
                operation = 'ins'
            else:
                if correct[x-1] == wrong[y-1]:
                    operation = 'copy'
                else:
                    operation = 'sub'
            if x > 1 and y > 1 and correct[x-1] == wrong[y-2] and correct[x-2] == wrong[y-1]:
                transchecklist = [min(minlist), d[x-2][y-2]+1]
                if transchecklist.index(min(transchecklist)) == 1:
                    operation = 'trans'

            # Then it is added to list with correct characters
            if operation == 'del':
                operations.append((operation, (' ' + correct)[x-1], (' ' + correct)[x]))
                x -= 1
            elif operation == 'ins':
                operations.append((operation, (' ' + wrong)[y-1], (' ' + wrong)[y]))
                y -= 1
            elif operation == 'copy':
                operations.append((operation, correct[x-1], correct[x-1]))
                x -= 1
                y -= 1
            elif operation == 'sub':
                operations.append(((operation, correct[x-1], wrong[y-1])))
                x -= 1
                y -= 1
            else:
                operations.append((operation, correct[x-2], correct[x-1]))
                x -= 2
                y -= 2
        operations.reverse()
        return(operations, distance)
    return distance

# Creating Confusion Matrices
ins = {}
dele = {}
sub = {}
trans = {}
print("Creating Confusion Matrices...")
# Loop over every line of spell-errors.txt
for i in range(len(errors)):
    if i%1000==0: print(str(i) + '/' + str(len(errors)) + ' errors')
    w, xlist = errors[i].split(":")
    xlist = xlist.strip().split(', ')
    for x in xlist:
        # Determining count of error
        if '*' in x:
            x, count = x.split('*')
            count = int(count)
        else:
            count = 1
        ops, _ = DLdistance(w, x)
        
        # Creating an element if it does not exist in the dictionary
        # Else adding count of it to the dictionary
        for op in ops:
            if op[0] == "ins":
                if not op[1]+op[2] in ins:
                    ins[op[1]+op[2]] = count
                else:
                    ins[op[1]+op[2]] += count
            elif op[0] == "del":
                if not op[1]+op[2] in dele:
                    dele[op[1]+op[2]] = count
                else:
                    dele[op[1]+op[2]] += count
            elif op[0] == "sub":
                if not op[1]+op[2] in sub:
                    sub[op[1]+op[2]] = count
                else:
                    sub[op[1]+op[2]] += count
            elif op[0] == "trans":
                if not op[1]+op[2] in trans:
                    trans[op[1]+op[2]] = count
                else:
                    trans[op[1]+op[2]] += count
print("Confusion matrices created.\n")


# Creating count dictionaries to calculate denominators
count1 = {}
count2 = {}
print("Calculating counts of occurences of characters in corpus...")

# For every element in matrices dictinary is looped over and count is added
# It is working but it can be improved to increase efficiency
for k,v in dele.items():
    if k not in count2:
        count2[k] = 0
    for k1,v1 in dictionary.items():
        if k in k1:
            count2[k] += v1
for k,v in trans.items():
    if k not in count2:
        count2[k] = 0
    for k1,v1 in dictionary.items():
        if k in k1:
            count2[k] += v1

for k,v in ins.items():
    if k[0] not in count1:
        count1[k[0]] = 0
    for k1,v1 in dictionary.items():
        if k[0] in k1:
            count1[k[0]] += v1
for k,v in sub.items():
    if k[1] not in count1:
        count1[k[1]] = 0
    for k1,v1 in dictionary.items():
        if k[1] in k1:
            count1[k[1]] += v1

# Explicitly searching for beginning of sentence characters
for word, count in dictionary.items():
    count1[' '] += count
    if not ' ' + word[0] in count2:
        count2[' '+word[0]] = count
    else:
        count2[' '+word[0]] += count
        
print("Counts calculated.\n")


# Predicting mispelled words
# Two different list for both with and without add-one smoothing
output = []
smoothed = []
print("Predicting misspelled words...")
import time
# Looping over every misspelled words
for i in range(len(wrong)):
    begi = time.time()
    print(str(i) + '/' + str(len(wrong))) if i%50==0 else 1
    wro = wrong[i]
    # Candidates list to store candidates with 1 edit distance to word
    candidates = []
    # Dictionary is searched to find candidates
    for word, prob in probs.items():
        if abs(len(word) - len(wro)) < 2: 
            ops, distance = DLdistance(word, wro)
            if distance == 1:
                # Candidates stored as tuples to store both word and operations
                candidates.append((word,ops))
    # If no candidates found, append empty string
    if len(candidates) == 0:
        output.append('')
        smoothed.append('')
        continue
        
    # Lists to store probabilities
    wordprobs = []
    addoneprobs = []
    for candidate in candidates:
        for op in candidate[1]:
            if op[0] == 'copy': continue
            
            # Channel model probabilities and add-one smoothed probabilities are 
            # calculated according to respective oepration
            elif op[0] == 'sub':
                if op[1]+op[2] in sub:
                    pofxgivenw = sub[op[1]+op[2]] / count1[op[2]]
                    addone = (sub[op[1]+op[2]] + 1) / (count1[op[2]] + 27)
                else:
                    pofxgivenw = 0
                    if op[2] in count1:
                        addone = 1 / (count1[op[2]] + 27)
                    else:
                        addone = 1 / 27
            elif op[0] == 'ins':
                if op[1]+op[2] in ins:
                    pofxgivenw = ins[op[1]+op[2]] / count1[op[1]]
                    addone = (ins[op[1]+op[2]] + 1) / (count1[op[1]] + 27)
                else:
                    pofxgivenw = 0
                    if op[1] in count1:
                        addone = 1 / (count1[op[1]] + 27)
                    else:
                        addone = 1 / 27
            elif op[0] == 'del':
                if op[1]+op[2] in dele:
                    pofxgivenw = dele[op[1]+op[2]] / count2[op[1]+op[2]]
                    addone = (dele[op[1]+op[2]] + 1) / (count2[op[1]+op[2]] + 27)
                else:
                    pofxgivenw = 0
                    if op[1]+op[2] in count2:
                        addone = 1 / (count2[op[1]+op[2]] + 27)
                    else:
                        addone = 1 / 27
            elif op[0] == 'trans':
                if op[1]+op[2] in trans:
                    pofxgivenw = trans[op[1]+op[2]] / count2[op[1]+op[2]]
                    addone = (trans[op[1]+op[2]] + 1) / (count2[op[1]+op[2]] + 27)
                else:
                    pofxgivenw = 0
                    if op[1]+op[2] in count2:
                        addone = 1 / (count2[op[1]+op[2]] + 27)
                    else:
                        addone = 1 / 27

        # P(w) is calculated
        pofw = probs[candidate[0]]
        
        # Probabilities append to the lists
        wordprobs.append(pofw * pofxgivenw)
        addoneprobs.append(pofw * addone)
    # Words with max probabilities appended to the output lists.
    output.append(candidates[wordprobs.index(max(wordprobs))][0])
    smoothed.append(candidates[addoneprobs.index(max(addoneprobs))][0])
    print(time.time() - begi)
print("All words predicted.\n")

print("Writing outputs to output.txt and addone.txt")
with open("output.txt", 'w') as f:
    for out in output:
        f.write(out + '\n')
        
with open("addone.txt", 'w') as f:
    for out in smoothed:
        f.write(out + '\n')    




# Alternative method that have been implemented by misunderstanding of cahnnel model probability.
# It calculates error probability on word level in spell-errors.txt given the correct word.
# I don't know it is a valuable or valid method but it performed better than channel model probability. 
# It only made one single error that the 'tiem' predicted as 'time' instead of 'item'.
# Since both words are very common, I suppose a human also couldn't correctly predict it all the time.

""" 
# Create an error dictionary where the structure is like following FOR OLD METHOD
# {correct_word : {error1 : error1_probability,
#                  error2 : error2_probability},
# ...
# }
errordict = {}
for error in errors:
    correct, error = error.split(":")
    error = error.strip().split(', ')
    errordict[correct] = {}
    total = 0
    for word in error:
        if '*' in word:
            word, count = word.split('*')
            errordict[correct][word] = int(count)
            total += int(count)
        else:
            errordict[correct][word] = 1
            total += 1
    for word, count in errordict[correct].items():
        errordict[correct][word] = count / total


# OLD METHOD
output = []
for i in range(len(wrong)):
    print(i) if i%10==0 else 1
    wro = wrong[i]
    candidates = []
    for word, prob in probs.items():
        if DLdistance(word, wro) == 1: candidates.append(word)
    if len(candidates) == 0:
        output.append('')
        continue
    wordprobs = []
    for candidate in candidates:
        if candidate in errordict:
            if wro in errordict[candidate]:
                pofw = probs[candidate]
                pofxgivenw = errordict[candidate][wro]
                wordprobs.append(pofw * pofxgivenw)
            else:
                wordprobs.append(0)
        else:
            wordprobs.append(0)
    output.append(candidates[wordprobs.index(max(wordprobs))])

 """


