from collections import Counter, namedtuple
from math import log
import nltk
import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, nminus1grams, ngrams, d, num_sent') # holds counts for the lm
DELIM = " " # delimiter for tokens in an ngram
count = 0 # number of sentences in training

def filter_text(file_name, text_filter):
    '''returns text without the labellers '''
    global count
    text = ""
    for i in open(file_name, "r").read().splitlines():
        if i.startswith(text_filter):
            count += 1
            text += i.replace(text_filter, '')
    return(text)

def generate_ngrams(tokens, n):
    """ Returns a list of ngrams made from a list of tokens """
    ngrams = []
    if n == 0:
        n = 1
    if n > 0:
        for i in range(0, len(tokens)-n+1):
            ngrams.append(DELIM.join(tokens[i:i+n]))

    return (Counter(ngrams))

def build_lm(file_name, n, threshold, category):
    """ Builds an ngram language model. """
    text = filter_text(file_name, category) 
    tokens = nltk.word_tokenize(text)
    nminus1grams = generate_ngrams(tokens, n - 1)
    ngrams = generate_ngrams(tokens, n)
    
    ## Calculate discounting factor
    n_1 = sum(x == 1 for x in ngrams.values())
    n_2 = sum(x == 2 for x in ngrams.values())
    d = n_1 / float(n_1 + 2 * n_2) 
    
    ## Position all ngrams below threshold into unknown
    n_threshold = sum(x for x in ngrams.values() if x <= threshold)
    ngrams = {k:v for k, v in ngrams.iteritems() if v > threshold}
    ngrams['<Unknown>'] = n_threshold

    n_threshold = sum(x for x in nminus1grams.values() if x <= threshold)
    nminus1grams = {k:v for k, v in nminus1grams.iteritems() if v > threshold}
    nminus1grams['<Unknown>'] = n_threshold

    global count
    temp = count
    count = 0

    return LanguageModel(len(tokens), set(tokens), nminus1grams, ngrams, d, temp)

def katz_backoff(lm, token, history = None):
    '''Returns the katz backoff probability ''' 
    ## Set history and token to <Unknown> if they cannot be found
    if history not in lm.nminus1grams.keys() and history != None:
        history = '<Unknown>'
    if token not in lm.nminus1grams.keys():
        token = '<Unknown>'
    prefix_count = lm.nminus1grams.get(history, 0)
    
    ## Calculate the probability for unigrams
    if history == None:
        if token == '<Unknown>':
            return 1.5 / float(lm.num_tokens)
        else:
            return lm.ngrams[token] / float(lm.num_tokens)

    ## calculate probability if ngrams can be found 
    if lm.ngrams.get(history+DELIM+token, 0) != 0:
        ngram_count = lm.ngrams.get(history + DELIM + token, 0) - lm.d
        return (ngram_count / float(prefix_count))
    else: 
        ## Calculate p_katz
        p_katz = 1 / float(lm.num_tokens)
        ## Calculate alpha
        seen = {k.split()[-1]: v for k, v in lm.ngrams.iteritems() if k.split()[0] == history and k.split()[0] != '<Unknown>'}
        unseen = {k: v for k, v in lm.nminus1grams.iteritems() if k not in seen.keys()}
        sum_p_star = (sum(seen.values()) - len(seen) * lm.d) / float(prefix_count)
        sum_p_katz = sum(unseen.values()) / float(lm.num_tokens)
        alpha = (1 - sum_p_star) / sum_p_katz
        return(alpha * p_katz)

    return sum_p_katz

def classifier(lm_r, lm_p, sent, n, Bayes = False):
    ''' return a string that indicates the class. classification is executed through
    iterating through the n-grams in order, calculating their probabilities through
    katz backoff and multiply the resulted probabilities'''
    tokens = nltk.word_tokenize(sent)
    prob_r = 1
    prob_p = 1
    for i in range(0, len(tokens)-n+1):
        if n == 1:
            prob_r *= katz_backoff(lm_r, tokens[i])
            prob_p *= katz_backoff(lm_p, tokens[i])

        else:
            prob_r *= katz_backoff(lm_r, tokens[i + 1], tokens[i])
            prob_p *= katz_backoff(lm_p, tokens[i + 1], tokens[i])
    if Bayes:
        prob_r *= lm_r.num_sent
        prob_p *= lm_p.num_sent
    if prob_r > prob_p: 
        return 'r'
    else:
        return 'p'

def evaluation(classified, positive, negative):

    '''this function calculates the precision, recall and f1 with respect to the categories'''

    tp = sum(v == positive  for a in classified for k in a.keys() for v in a[k] if k == positive)
    fp = sum(v == positive  for a in classified for k in a.keys() for v in a[k] if k == negative)
    fn = sum(v == negative  for a in classified for k in a.keys() for v in a[k] if k == positive)
    
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    return {"Precision":precision, "Recall": recall, "F1" : 2 * precision * recall / (precision + recall)}



if __name__ == '__main__':

    ## Unigrams
    
    lm_review = build_lm("hw2_train.txt", 1, 1, "r: ") 
    lm_plot = build_lm("hw2_train.txt", 1, 1, "p: ")

    classified = []
    for i in open("hw2_test.txt", "r").read().splitlines():
        correct = i[0]
        classified.append({correct : classifier(lm_review, lm_plot, ' '.join(i.split()[1:]), 1)})
    
    print('For Unigram language models')
    print('stats with regard to p')
    print(evaluation(classified, 'p', 'r'))
    print('stats with regard to r')
    print(evaluation(classified, 'r', 'p'))

    classified = []
    for i in open("hw2_test.txt", "r").read().splitlines():
        correct = i[0]
        classified.append({correct : classifier(lm_review, lm_plot, ' '.join(i.split()[1:]), 1, True)})
    print('With Bayes')
    print('stats with regard to p: ')
    print(evaluation(classified, 'p', 'r'))
    print('stats with regard to r')
    print(evaluation(classified, 'r', 'p'))
    
    ## Bigrams

    lm_review = build_lm("hw2_train.txt", 2, 1, "r: ") 
    lm_plot = build_lm("hw2_train.txt", 2, 1, "p: ")

    classified = []
    for i in open("hw2_test.txt", "r").read().splitlines():
        correct = i[0]
        a = {correct : classifier(lm_review, lm_plot, ' '.join(i.split()[1:]), 2)}
        classified.append(a)
    
    print('For bigram language models')
    print('stats with regard to p')    
    print(evaluation(classified, 'p', 'r'))
    print('stats with regard to r')    
    print(evaluation(classified, 'r', 'p'))
    
    for i in open("hw2_test.txt", "r").read().splitlines():
        correct = i[0]
        classified.append({correct : classifier(lm_review, lm_plot, ' '.join(i.split()[1:]), 2, True)})
    
    print('With Bayes')
    print('stats with regard to p')    
    print(evaluation(classified, 'p', 'r'))
    print('stats with regard to r')
    print(evaluation(classified, 'r', 'p'))

    






    
    
