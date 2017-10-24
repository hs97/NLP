from collections import namedtuple, Counter
from itertools import product, combinations
from operator import mul
import pandas as pd

POS = namedtuple('POS', 'trans_count, trans_prob, emission_count, emission_prob')

def build_tags(text):
    '''Build a named tuple that contains emission and transistion stats''' 
    
    ## Attain observations, tags and vocab from the corpus through list comprehension
    obs = {phrase.split('/')[1]:[string.split('/')[0] for sent in text for string in sent.split() if string.split('/')[1] == phrase.split('/')[1]] for sent in text for phrase in sent.split()}
    tags = obs.keys() + ['START', 'END']
    vocab = list({string.split('/')[0] for sent in text for string in sent.split()})

    ## Attain emission count with add-one smoothing
    add_one_emission = {tag:vocab for tag in obs.keys()}
    emission_count = {tag:Counter(obs[tag] + add_one_emission[tag]) for tag in obs.keys()}

    tags_count = {k:len(tags) for k in tags}
    trans_count = {k:{k:1 for k in tags if k != 'START'} for k in tags if k != 'END'}
    for sent in text:
        tags_sent = [string.split("/")[1] for string in sent.split()]
        tags_sent = ['START'] + tags_sent + ['END']
        tags_sent = zip(tags_sent, tags_sent[1:])
        for k in tags_sent:
            trans_count[k[0]][k[1]] += 1
            tags_count[k[0]] += 1
    
    emission_prob = {k:{k2:v2/float(sum(v.values())) for k2, v2 in v.iteritems()} for k, v in emission_count.iteritems()}
    trans_prob = {k:{k2:v2/float(sum(v.values())) for k2, v2 in v.iteritems()} for k, v in trans_count.iteritems()}
    
    return POS(trans_count, trans_prob, emission_count, emission_prob)

def joint_prob(sent, tags, POS):
    ''' returns the joint probability'''
    sent = sent.split()
    tags = tags.split()
    ## find emission probability for word/tag pairs
    emission_prob = [POS.emission_prob[string[1]][string[0]] for string in zip(sent, tags)]
    tags = ['START'] + tags + ['END']
    tags = zip(tags, tags[1:])
    ## find transistion probability for tag/prev_tag pairs
    trans_prob = [POS.trans_prob[k[0]][k[1]] for k in tags]
    return reduce(mul, emission_prob) * reduce(mul, trans_prob)

def viterbi(sent, POS):
    ''' Return a viterbi table and a tag sequence '''
    ## Initialize T, N, viterbi table and backpointer, initialize start column with probability 1
    T = ['START'] + sent.split() + ['END']
    N = POS.emission_prob.keys()
    viterbi = {k:{k:1 for k in N} for k in T}
    backpointer = {k:{k:0 for k in N} for k in T}
    
    ## Calculate probability for for column
    for state in N:
        viterbi[T[1]][state] = POS.trans_prob[T[0]][state] * POS.emission_prob[state][T[1]]
        backpointer[T[1]][state] = 0

    ## Recursive step
    for t in range(2, len(T)):
        for state in N:            
            if t != len(T) - 1:
                prev_state = {s:(viterbi[T[t - 1]][s] * POS.trans_prob[s][state]) for s in N}
                backpointer[T[t]][state] = max(prev_state, key = prev_state.get) 
                viterbi[T[t]][state] = max(prev_state.values()) * POS.emission_prob[state][T[t]]
            else: 
                viterbi[T[t]][state] = viterbi[T[t - 1]][state] * POS.trans_prob[state]['END']
    
    ## Find the max end state and trace back the tag sequence
    end = {max(viterbi['END'], key = viterbi['END'].get):max(viterbi['END'].values())}
    last_tag = end.keys()[0]
    tag_sequence = [last_tag]
    for i in range(len(T) - 2, 1, -1):
        last_tag = backpointer[T[i]][last_tag]
        tag_sequence = [last_tag] + tag_sequence
        
    return [viterbi, tag_sequence]


text = open("corpus.txt", "r").read().splitlines()
POS = build_tags(text)

print("Question 1.1")
print(pd.DataFrame(POS.trans_count))

print("Question 1.2")
print(pd.DataFrame(POS.trans_prob))

print("Question 1.3")
print(pd.DataFrame(POS.emission_count))

print("Question 1.4")
print(pd.DataFrame(POS.emission_prob))

sent = 'show your light when nothing is shining'

tags = 'NOUN PRON NOUN ADV NOUN VERB NOUN'
print("Question 2.1.a)")
print(joint_prob(sent, tags, POS))

tags = 'VERB PRON NOUN ADV NOUN VERB VERB'
print("Question 2.1.b")
print(joint_prob(sent, tags, POS))

tags = 'VERB PRON NOUN ADV NOUN VERB NOUN'
print("Question 2.1.c")
print(joint_prob(sent, tags, POS))

print("Question 2.2")
print(pd.DataFrame(viterbi(sent, POS)[0]))

print("Tag sequence")
print(viterbi(sent, POS)[1])

tags = 'PREP DET NOUN CONJ PRON VERB VERB'
print("End State Probability is:")
print(joint_prob(sent, tags, POS))


