from token_class import Tokens, parse_sentence
from libs.stanford_corenlp_pywrapper import CoreNLP
coreNlpPath = "corenlp/*"
proc = CoreNLP("parse", corenlp_jars=[coreNlpPath])

from definitions import Sentence_props

def find_indices_events(words, e1, e2):
    return words.index(e1), words.index(e2)

same_sent = open('same_Sentence_relation_2.txt', 'r')
all_data = []
i = 0
t = 0
for line in same_sent:
    i += 1
    print i
    temp = line.strip().split('\t\t')
    lemmas = []
    pos_tags = []
    word_forms = []
    parse = []
    sent = temp[0]
    stanford_parse = proc.parse_doc(sent)
    if len(stanford_parse['sentences']) > 1:
        t += len(stanford_parse['sentences'])
        print len(stanford_parse['sentences'])
        print line
    for sentence in stanford_parse['sentences']:
        lemmas += sentence["lemmas"]
        pos_tags += sentence['pos']
        word_forms += sentence['tokens']
        parse += sentence['deps_cc']
        i1, i2 = find_indices_events(word_forms, temp[1], temp[2])
        #print lemmas, pos_tags, word_forms, parse, i1, i2, temp[1], temp[2], temp[3]
        all_data.append(Sentence_props(lemmas, pos_tags, word_forms, parse, i1, i2, temp[1], temp[2], temp[3]))
print t
import pickle
with open('temprel_data', 'wb') as fp:
    pickle.dump(all_data, fp)