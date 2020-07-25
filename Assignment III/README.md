
# Assignment III - Extractive Text Summarization for COVID-19
In this project I have implemented a text summarization model similar to LexRank, using PageRank algorithm. More information about LexRank can be found [here](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html).

## Files
`04-10-mag-mapping.csv` file maps the abstract of the papers to their ids.

`topics-rnd1.xml` file lists the topics.

`qrels-rnd1.txt` file shows the relation score of a paper to a specific topic. First column specifies the topic id, third column specifies the paper id and last column specifies the relation score in the range of [0-2].

All files are obtained from the Round 1 Data section of this [website](https://ir.nist.gov/covidSubmit/data.html). 10 April version of data used in this project. 

## Model
Only abstract of articles with relevance score of 2 with the respective topic are used while creating this model. Among those documents top 10 most relevant documents are identified using PageRank algorithm. Document graph between documents is constructed, where each node corresponds to a document and each edge represents the TF-IDF weighted cosine similarity between the abstracts of the corresponding two documents.

Among 10 documents top 20 sentences are identified using similar method. Similar to document graph sentence graph is constructed among sentences of top 10 documents, where each node corresponds to a sentence and each edge represents the TF-IDF weighted cosine similarity between corresponding two sentences.

These 20 sentences are the summary of given topic.

I have extracted summaries for 3 topics:

 - Topic 1, Coronavirus Origin
 - Topic 13, How Does Coronavirus Spread
 - Topic 23, Coronavirus and Hypertension 

## Execution of the Program
I have implemented the program on python3. Mapping and the relations file must be present near to the script.
```bash
python3 summarization.py
```
Program outputs the results to the same directory.

Detailed information and results can be found on the report.
