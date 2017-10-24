import xml.etree.ElementTree as ET
from collections import Counter
import nltk
import collections
from nltk.metrics.scores import precision, recall, f_measure
from operator import is_not
from functools import partial

def parse_xml(file_name):
    '''parse the xml and return the data'''
    tree = ET.parse(file_name)
    root = tree.getroot()
    label = []
    word = []
    pos = []
    for answer in root.iter('answer'):
        label.append(answer.attrib.values()[1])
    for context in root.iter('context'):
        pos.append([e.attrib['pos'] for e in context])
        word.append(context.text.split() + [e.tail for e in context][:-1])
    word_pos = [zip(w, p) for w in word for p in pos]
    return {"word":word, "pos":pos, "label":label, "word_pos":word_pos}

def window(x, indices, window_size): 
    '''pick the window according to window size'''
    windows = [context[index - window_size:index] + context[index + 1:index + window_size + 1]
            for context, index in zip(x, indices)]
    return windows 

def get_bag(x, threshold, threshold_2):
    '''get the bag of words'''
    bag_of_words = sum([Counter(word) for word in x["word"]], Counter())
    bag_of_words = [key for key, value in bag_of_words.iteritems() if value >= threshold]
    bag_of_pos = sum([Counter(pos) for pos in x["pos"]], Counter())
    bag_of_pos = [key for key, value in bag_of_pos.iteritems() if value >= threshold_2]
    return {"bow":bag_of_words, "bop":bag_of_pos}

def features(x, bag, window_size):
    '''get the 6 features'''
    ## collocation
    indices = [context.index(each) for context in x["word"] for each in context if '  ' in each]
    window_index = range(-window_size, 0) + range(1, window_size + 1)
    collocation_word = [{("word" + str(i)): word for word, i in zip(windows, window_index)} for windows in 
            window(x["word"], indices, window_size)]
    collocation_pos = [{("pos" + str(i)): word for word, i in zip(windows, window_index)} for windows in 
            window(x["pos"], indices, window_size)]
    collocation_both = [dict(word, **pos)for word, pos in zip(collocation_word, collocation_pos)]
    
    ## co-occurence
    cooccurence_word = [{word:(1 if word in context.values() else 0) for word in bag["bow"]} for context in collocation_word]
    cooccurence_pos = [{word:(1 if word in context.values() else 0) for word in bag["bop"]} for context in collocation_pos]
    cooccurence_both = [dict(word, **pos) for word, pos in zip(cooccurence_word, cooccurence_pos)]
    
    ## both
    feature_word = [dict(collocation, **cooccurence) for collocation, cooccurence in zip(collocation_word, cooccurence_word)]
    feature_both = [dict(collocation, **cooccurence) for collocation, cooccurence in zip(collocation_both, cooccurence_both)]

    return({"cl_word":collocation_word, "cl_both":collocation_both, "co_word":cooccurence_word, "co_both":cooccurence_both, 
        "ft_word":feature_word, "ft_both":feature_both})

def scores(classifier, test, ids):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    accuracy = nltk.classify.accuracy(classifier, test)
    print("accuracy: " + str(accuracy))
    p = filter(partial(is_not, None), [precision(refsets[sense], testsets[sense]) for sense in ids])
    p = sum(p) / len(p)
    print("precision: " + str(p))
    r = filter(partial(is_not, None), [recall(refsets[sense], testsets[sense]) for sense in ids])
    r = sum(r) / len(r)
    print("recall: " + str(r) )
    f_1 = filter(partial(is_not, None), [f_measure(refsets[sense], testsets[sense]) for sense in ids]) 
    f_1 = sum(f_1) / len(f_1)
    print("f-1 score: " + str(f_1))

    return({"precision":p, "recall":r, "f_1":f_1, "accuracy":accuracy})

train = (parse_xml('bank.ntrain.xml'))
test = (parse_xml('bank.ntest.xml'))
bag = get_bag(train, 10, 20)
features_train = features(train, bag, 4)
features_test = features(test, bag, 4)

ids = list(set(test["label"]))

cl_word = [(word, label) for label, word in zip(train["label"], features_train["cl_word"])]
cl_word_test = [(word, label) for label, word in zip(test["label"], features_test["cl_word"])]
classifier = nltk.NaiveBayesClassifier.train(cl_word)
print("For collocation with words: ")
scores(classifier, cl_word_test, ids)
print("")


cl_both = [(word, label) for label, word in zip(train["label"], features_train["cl_both"])]
cl_both_test = [(word, label) for label, word in zip(test["label"], features_train["cl_both"])]
classifier = nltk.NaiveBayesClassifier.train(cl_both)
print("For collocation with words and pos: ")
scores(classifier, cl_both_test, ids)
print("")


co_word = [(word, label) for label, word in zip(train["label"], features_train["co_word"])]
co_word_test = [(word, label) for label, word in zip(test["label"], features_test["co_word"])]
classifier = nltk.NaiveBayesClassifier.train(co_word)
print("For coccurence with words: ")
scores(classifier, co_word_test, ids)
print("")

co_both = [(word, label) for label, word in zip(train["label"], features_train["co_both"])]
co_both_test = [(word, label) for label, word in zip(test["label"], features_test["co_both"])]
classifier = nltk.NaiveBayesClassifier.train(co_both)
print("For coccurence with words and pos: ")
scores(classifier, co_both_test, ids)
print("")

ft_word = [(word, label) for label, word in zip(train["label"], features_train["ft_word"])]
ft_word_test = [(word, label) for label, word in zip(test["label"], features_test["ft_word"])]
classifier = nltk.NaiveBayesClassifier.train(ft_word)
print("For both with words: ")
scores(classifier, ft_word_test, ids)
print("")

ft_both = [(word, label) for label, word in zip(train["label"], features_train["ft_both"])]
ft_both_test = [(word, label) for label, word in zip(test["label"], features_test["ft_both"])]
classifier = nltk.NaiveBayesClassifier.train(ft_both)
print("For both with words and pos: ")
scores(classifier, ft_both_test, ids)
print("")









