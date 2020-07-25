# Assignment I - Spelling Corrector
In this project I have implemented an isolated word spelling error corrector based on the noisy channel model.

## Files
Dictionary is created from the `corpus.txt` file obtained from Peter Norvig's [website](http://norvig.com/big.txt). While building the model I assumed the words inside corpus are spelled correctly.

While estimating error probability I have also used `spell-errors.txt` file obtained from Peter Norvig's [website](http://norvig.com/ngrams/spell-errors.txt). Each line of the file contains a correct word followed by a column and a comma and space separated list of observed misspelled versions of the word. **x* denotes that the prior word has been observed *x* number of times.

`test-words-misspelled.txt` and `test-words-correct.txt` files contain 384 misspelled and its corresponding correct words that are used to test the program.

## Model
I have implemented noisy channel model using above files. Definition and more information about the model can be found [here](https://web.stanford.edu/~jurafsky/slp3/B.pdf).

## Execution of the Program
I have implemented the noisy channel model on python3. All above files must be in the same directory as the script.
```bash
python3 spellCorrector.py
```
Detailed information and results can be found on the report.
