inverse_relation_map = {
    'AFTER':'BEFORE',
    'BEFORE':'AFTER',
    'SIMULTANEOUS':'SIMULTANEOUS',
    'IBEFORE':'IAFTER',
    'IAFTER':'IBEFORE',
    'IS_INCLUDED':'INCLUDES',
    'INCLUDES':'IS_INCLUDED',
    'IDENTITY':'IDENTITY',
    'BEGUN_BY':'ENDED_BY',
    'ENDED_BY':'BEGUN_BY',
    'BEGINS':'ENDS',
    'ENDS':'BEGINS',
    'DURING_INV':'DURING',
    'DURING':'DURING_INV',
}

class Sentence_props(object):
    def __init__(self, lemmas, pos, words, parse, i1, i2, e1, e2, relation):
        self.words = {}
        for i in range(len(lemmas)):
            self.words[i] = Nodes(lemmas[i], pos[i], words[i])
        for p in parse:
            if p[1] != p[2]:
                self.words[p[2]].parent = (p[1], p[0])
                if p[1] != -1:
                    self.words[p[1]].children += [(p[2], p[0])]

        self.parse = parse
        if i1 < i2:
            self.i1 = i1
            self.i2 = i2
            self.e1 = e1
            self.e2 = e2
            self.relation = relation
        else:
            self.i1 = i2
            self.i2 = i1
            self.e1 = e2
            self.e2 = e1
            self.relation = inverse_relation_map[relation]

class Nodes(object):
    def __init__(self, lemma, pos, word_form):
        self.lemma = lemma
        self.pos = pos
        self.word_form = word_form
        self.parent = (-1, 'root')
        self.children = []