import math 
import nltk
import operator
import random
from itertools import chain
from itertools import repeat
from collections import Counter

## This function allows the user to read in all the n_grams in a 
## corpus. 
def n_gram_reader(n, text, add_one = True): 
    
    ## Minor sanity check
    if n <= 0:
        sys.exit("invalid n")

    ## Zip allows for element-wise combination.
    ## This way creates lists of n-words that are 
    ## adjacent to each other
    n_gram_read = zip(* [text[i:] for i in range(n)]) 
    
    ## Concatenate the list into strings for better data 
    ## manipulation
    for i in range(len(n_gram_read)):
        n_gram_read[i] = ' '.join(n_gram_read[i])
    
    ## Give user the option to use add-1 counting or not
    ## Setting add_one is False is better for manipulation in
    ## the generator function 
    if add_one:
        n_gram_count = Counter(n_gram_read + list(set(n_gram_read)))
    else:
         n_gram_count = Counter(n_gram_read)

    return(n_gram_count)

## This function allows users to compute the log probabilities of n-grams
def log_prob(n, text): 

    ## Minor sanity check
    if n <= 0:
        sys.exit("invalid n")

    ## read in n_gram for level(n-1) and level(n)
    n_gram_count = n_gram_reader(n, text)
    if(n == 1):
        n_gram_prev = n_gram_count
    else:
        n_gram_prev = n_gram_reader(n - 1, text)

    word_type = len(set(text))

    ## check for number of total appearance for a n-gram
    ## by looking up what's in the level above
    for key in n_gram_count.keys():
        if n == 1: 
            prev = key
        else: 
             prev = ' '.join(key.split()[:n-1])

        ## compute log probability
        n_gram_count[key] = math.log(n_gram_count[key] / float(n_gram_prev[prev] - 1 + word_type))
    return(n_gram_count)

## This function allows users to print out number of answers in a given order

def print_top(x, n, order):
    x = sorted(x.items(), key = operator.itemgetter(1), reverse = (order == "descending"))
    for i in range(n):
        print(x[i])
    return

## This function generates text randomly by using the n-gram model and add-one smoothing

def generate(n, text):
    n_gram = n_gram_reader(n, text, False)

    ## If n is equal to 1, we don't filter based on '<s>'
    if n == 1:
        sent = '<s>'
        last = []

    ## Otherwise, filter for n-grams starting with '<s>' and pick them based on 
    ## the count for their appearance randomly
    ## Note: using split allows for easier filtering and token manipulation
    else:
        this_gram = {k:v for k, v in n_gram.items() if k.split()[0] == '<s>'}
        this_gram = (list(repeat(k, v)) for k, v in this_gram.items())
        this_gram = [item for sub in this_gram for item in sub]
        sent = random.choice(this_gram)
        last = sent.split()[1:n]

    ## while statement checks for whether '</s>' has been generated and sets
    ## a threshold of 20 words. 
    while '</s>' not in last and len(sent.split()) <= 20: 
        
        ## this statement filters the phrases that contains 'last' and pick the last word. We then generate
        ## a list in which a word appears however many times it was counted in n_gram. Then we add the vocabulary
        ## in the corpus into this list to complete add-one smoothing
        this_gram = {' '.join(k.split()[n-1:n]):v for k, v in n_gram.items() if k.split()[:n-1] == last or len(k.split()) == 1}  ## len(k.split()) == 1 ensures handling of n=1
        this_gram = (list(repeat(k, v)) for k, v in this_gram.items()) 
        this_gram = [item for sub in this_gram for item in sub] + list(set(text))
        this_gram = random.choice(this_gram)

        ## update 'last' as well as the sentence 
        last = last[1:n]
        last.append(this_gram)
        sent = sent + ' ' + this_gram

    ## check if threshold is reached or '</s> is generated'
    if sent.split()[-1] != '</s>':
        sent = sent + ' </s>'
    return(sent)

     
     
     
     
## Question 2.1

text = open("Q1_corpus.txt", "r").read().split()
print("\nQuestion 2.1:")
print_top(x = n_gram_reader(2, text), n = 10, order = "descending")

## Question 2.2

text = open("Q1_corpus.txt", "r").read().split()
print("\nQuestion 2.2:")
print_top(x = log_prob(2, text), n = 10, order = "descending")

## Question 2.3

file = "moviereview.txt"
text = ""

## append sentence symbols in front and at the end of sentences
for i in open(file, "r").read().splitlines():
    text = text + "<s> " + i + "</s> "

## lower-casing all the text 
text = text.lower().split()
print("\nQuestion 2.3:")
print_top(x = log_prob(2, text), n = 20, order = "descending")

## Question 2.4

text = open("bronte_jane_eyre.txt", "r").read()

## uses nltk sentence tokenizer to tokenize text
sent_token = nltk.sent_tokenize(text)
text = []

## append '<s>' and '</s>' around sentences
## then use word tokenizer 
for sent in sent_token:
    sent = nltk.word_tokenize(sent)
    sent.append('</s>')
    sent = ['<s>'] + sent
    text.append(sent)
text = list(chain(*text))
print("\nQuestion 2.4:")
for i in range(5):
    print("Generated by " + str(i + 1) + "-gram")
    print(generate(i + 1, text))


