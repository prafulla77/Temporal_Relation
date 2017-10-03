from definitions import Sentence_props
#pos vector dimension = 28
#maximum sequence size = 40
rule4 = 0

class Features(object):
    def __init__(self, sentence_prop):
        self.sentence_prop = sentence_prop
        temp_path = self.get_deprel_path()
        # dependency
        self.dependency_path = []
        for elem in temp_path:
            self.dependency_path.append(self.sentence_prop.words[elem].parent[1])
        #dependency
        for elem in [self.sentence_prop.i1, self.sentence_prop.i2]:#temp_path: #change it to temp_path or self.dependency_path to test variations
            temp_path += self.get_local_context(elem)
        #temp_path += self.get_prep_feat(self.sentence_prop.i1)
        #temp_path += self.get_prep_feat(self.sentence_prop.i2)
        self.path = sorted(set(temp_path))

    def get_trigger_feats(self):
        i1 = self.sentence_prop.i1
        i2 = self.sentence_prop.i2
        return (self.sentence_prop.words[i1].word_form, self.sentence_prop.words[i1].word_form,
                self.sentence_prop.words[i1].pos, self.sentence_prop.words[i1].pos)


    def get_deprel_path(self):
        i1 = self.sentence_prop.i1
        i2 = self.sentence_prop.i2
        graph = self.sentence_prop.words
        visited = [False]*len(graph)
        stack = [(i1,0)]
        visited[i1] = True
        path = []
        while len(stack) > 0 and stack[-1][0] != i2:
            pop = True
            prev_height = stack[-1][1]
            neighbors = graph[stack[-1][0]].children + [graph[stack[-1][0]].parent]
            for node in neighbors:
                if not visited[node[0]] and node[0] != -1:
                    stack += [(node[0], prev_height+1)]
                    visited[node[0]] = True
                    pop = False
            if pop:
                del stack[-1]
        if stack[-1][0] == i2:
            path = [i2]
            last_depth = stack[-1][1]
        while(len(stack) > 0):
            if stack[-1][1] == last_depth:
                del stack[-1]
            else:
                last_depth = stack[-1][1]
                path = [stack[-1][0]] + path
        return path

    def get_local_context(self, index):
        path = []
        for child in self.sentence_prop.words[index].children:
            if child[1] in ['mark', 'nmod:tmod', 'case', 'aux', 'conj', 'expl', 'cc', 'cop', 'amod', 'advmod', 'punct', 'ref']: #Rules 2 & 3 (assuming punct as part of dependency children)
                path.append(child[0])
        return path

    def get_prep_feat(self, index): #only e1 and e2 index
        global rule4
        path = []
        for child in self.sentence_prop.words[index].children:
            if child[1].split(':')[0].lower() == 'nmod' and 'NNP' not in self.sentence_prop.words[child[0]].pos: #Rules 2 & 3 (assuming punct as part of dependency children)
                #path.append(child[0])
                to_add = False
                for grand_child in self.sentence_prop.words[child[0]].children:
                    if grand_child[1] == 'case':
                        to_add = True
                        path.append(grand_child[0])
                if to_add:
                    rule4 += 1
                    path.append(child[0])
        return path

    def get_pos_features(self):
        pos_sequence = []
        for index in self.path:
            '''
            if 'NN' in self.sentence_prop.words[index].pos:
                pos_sequence.append('NN')
            else:
            '''
            pos_sequence.append(self.sentence_prop.words[index].pos)
        return pos_sequence

    def get_word_features(self):
        word_sequence = []
        for index in self.path:
            word_sequence.append(self.sentence_prop.words[index].word_form)
        return word_sequence

    def get_dep_features(self):
        dep_sequence = []
        for index in self.path:
            dep_sequence.append(self.sentence_prop.words[index].parent[1])
        #print dep_sequence, self.dependency_path
        return dep_sequence

    def get_relation(self):
        return self.sentence_prop.relation

import pickle, numpy as np
pos_padding = [0.01]*38

out_map_14 = {
    'BEFORE':[1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'AFTER':[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'SIMULTANEOUS':[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'IBEFORE':[0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'IAFTER':[0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'IS_INCLUDED':[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'INCLUDES':[0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'IDENTITY':[0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'BEGUN_BY':[0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'ENDED_BY':[0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'BEGINS':[0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'ENDS':[0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'DURING':[0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'DURING_INV':[0,0,0,0,0,0,0,0,0,0,0,0,0,1],
}

out_map_3 = {
    'BEFORE':[1,0,0],
    'AFTER':[0,1,0],
    'SIMULTANEOUS':[0,0,1],
    'IBEFORE':[1,0,0],
    'IAFTER':[0,1,0],
    'IS_INCLUDED':[0,0,1],
    'INCLUDES':[0,0,1],
    'IDENTITY':[0,0,1],
    'BEGUN_BY':[0,0,1],
    'ENDED_BY':[0,0,1],
    'BEGINS':[0,0,1],
    'ENDS':[0,0,1],
    'DURING':[0,0,1],
    'DURING_INV':[0,0,1],
}

with open('vocab/pos.vector', 'rb') as fp:
    pos_vec = pickle.load(fp)

word_vec = {}
fp = open('vocab/deps.words', 'r')
for l in fp:
    t = l.strip().split()
    word_vec[t[0].lower()] = [float(x) for x in t[1:]]

fp = open('vocab/padding_unknown_300d.txt', 'r')
for l in fp:
    t = l.strip().split()
    word_vec[t[0]] = [float(x) for x in t[1:]]


with open('vocab/deps.vector', 'rb') as fp:
    dep_rel = pickle.load(fp)

def get_train_features(train_data):
    maxim = 0
    X1, X2, X3, X4, X5, X6, X7, X8 = [], [], [], [], [], [], [], []
    Y = []
    for elem in train_data:
        F = Features(elem)
        temp = F.get_pos_features()
        temp_len = len(temp)
        maxim = max(temp_len, maxim)
        ip_feat_1 = []
        ip_feat_2 = []
        for pos in temp:
            ip_feat_1.append(pos_vec.get(pos, pos_vec['UNKNOWN']))
            ip_feat_2 = [pos_vec.get(pos, pos_vec['UNKNOWN'])]+ip_feat_2
        ip_feat_1 += [pos_padding]*(16-temp_len)
        ip_feat_2 += [pos_padding]*(16-temp_len)
        X1.append(ip_feat_1)
        X2.append(ip_feat_2)

        temp = F.get_word_features()
        temp_len = len(temp)
        ip_feat_1 = []
        ip_feat_2 = []
        for word in temp:
            ip_feat_1.append(word_vec.get(word.lower(), word_vec['UNKNOWN']))
            ip_feat_2 = [word_vec.get(word.lower(), word_vec['UNKNOWN'])]+ip_feat_2
        ip_feat_1 += [word_vec['PADDING']]*(16-temp_len)
        ip_feat_2 += [word_vec['PADDING']]*(16-temp_len)
        X3.append(ip_feat_1)
        X4.append(ip_feat_2)

        temp = F.dependency_path #F.get_dep_features()#
        temp_len = len(temp)
        ip_feat_1 = []
        ip_feat_2 = []
        for dep in temp:
            ip_feat_1.append(dep_rel.get(dep, dep_rel['UNKNOWN']))
            ip_feat_2 = [dep_rel.get(dep, dep_rel['UNKNOWN'])] + ip_feat_2
        ip_feat_1 += [dep_rel['PADDING']] * (12 - temp_len)
        ip_feat_2 += [dep_rel['PADDING']] * (12 - temp_len)
        # print len(ip_feat_2), len(ip_feat_1)
        X5.append(ip_feat_1)
        X6.append(ip_feat_2)

        w1,w2,p1,p2 = F.get_trigger_feats()
        X7.append(word_vec.get(w1.lower(), word_vec['UNKNOWN']) + word_vec.get(w2.lower(), word_vec['UNKNOWN']))
        X8.append(pos_vec.get(p1,pos_vec['NN']) + pos_vec.get(p2,pos_vec['NN']))
        Y.append(out_map_14[F.get_relation()])
    X1 = np.asarray(X1)
    print X1.shape, maxim
    X2 = np.asarray(X2)
    X3 = np.asarray(X3)
    X4 = np.asarray(X4)
    X5 = np.asarray(X5)
    X6 = np.asarray(X6)
    X7 = np.asarray(X7)
    X8 = np.asarray(X8)
    Y = np.asarray(Y)
    print X1.shape, X2.shape, Y.shape, X6.shape, "Rule 4 ===  ",rule4
    #return [X1,X2,X3,X4,X5,X6], Y
    #exit()
    return [X1, X2, X3, X4, X5, X6], Y


