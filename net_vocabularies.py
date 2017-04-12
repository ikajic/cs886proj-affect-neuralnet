import os
import re
from nengo import spa
import pandas as pd


def create_spa_vocabularies(D):
    spa_vocabs = {}

    identities = dict(MOTHER='0.4825*E+0.375*P+0.0725*A',
                      CHILD='0.425*E-0.4275*P+0.6875*A',
                      GIRLFRIEND='0.6175*E+0.18*P+0.56*A',
                      BOYFRIEND='0.27*E+0.1925*P+0.3375*A'
                      )

    verbs = dict(ABANDON='-0.485*E-0.0125*P-0.035*A',
                 HUG='0.5825*E-0.22*P-0.0925*A',
                 CHEAT='-0.43*E+0.1925*P+0.165*A',
                 LIE_TO='-0.5425*E-0.135*P+0.0275*A',
                 )

    emotions = dict(ANGRY="-0.4675*E+-0.0375*P+-0.035*A",
                    DISGUSTED="-0.3625*E+-0.1825*P+-0.0625*A",
                    FEARFUL="-0.535*E+-0.4375*P+-0.0275*A",
                    HAPPY="0.6175*E+0.3125*P+0.2425*A",
                    SAD="-0.47*E+-0.365*P+-0.3125*A",
                    SURPRISED="0.295*E+0.11*P+0.2375*A")

    vocab = dict(pos=['SUBJECT', 'VERB', 'OBJECT'],
                 verbs=verbs,
                 identities=identities,
                 emotions=emotions)

    spa_vocabs = {}

    spa_vocabs['all'] = spa.Vocabulary(dimensions=D)
    spa_vocabs['epa'] = spa.Vocabulary(dimensions=D)
    spa_vocabs['epa'].parse('E+P+A')
    spa_vocabs['all'].parse('SUBJECT+VERB+OBJECT')

    for category, items in vocab.iteritems():
        spa_vocabs[category] = spa.Vocabulary(dimensions=D)
        if isinstance(items, dict):
            for word, vector in items.iteritems():
                spa_vocabs[category].parse(word)
                spa_vocabs['all'].add(word, spa_vocabs[category][word].v)

    return vocab, spa_vocabs


def clean_word(w):
    word = re.sub('[^0-9a-zA-Z]+', '_', w)
    word = word.upper()
    return word


def create_spa_vocabularies_whole(D, n=1000):
    spa_vocabs = {}
    path = './database/'
    vocab = {}
    labels = ['e1', 'p1', 'a1', 'e2', 'p2', 'a2']

    vocab['id'] = pd.read_csv(
        os.path.join(path, 'identities.txt'), header=None)

    vocab['id'].columns = ['word', 'e1', 'p1', 'a1', 'e2', 'p2', 'a2', 'tmp']

    vocab['verb'] = pd.read_csv(
        os.path.join(path, 'behaviors.txt'), header=None)
    vocab['verb'].columns = ['word', 'e1', 'p1', 'a1', 'e2', 'p2', 'a2', 'tmp']

    for v in vocab:
        # normalize to [-1,1] range
        vocab[v][labels] = vocab[v][labels].apply(lambda x: x/4.)
        vocab[v][labels] = vocab[v][labels].clip(-1, None)

    identities = {}
    for i, row in vocab['id'][:n].iterrows():
        word = clean_word(row.word)
        e, p, a, = row['e1'], row['p1'], row['a1']
        expr = ("%.2f*E+%.2f*P+%.2f*A" % (e, p, a)).replace("+-", "-")
        identities[word] = expr

    verbs = {}
    for i, row in vocab['verb'][:n].iterrows():
        word = clean_word(row.word)
        e, p, a, = row['e1'], row['p1'], row['a1']
        expr = ("%.2f*E+%.2f*P+%.2f*A" % (e, p, a)).replace("+-", "-")
        verbs[word] = expr

    emotions = dict(ANGRY="-0.4675*E+-0.0375*P+-0.035*A",
                    DISGUSTED="-0.3625*E+-0.1825*P+-0.0625*A",
                    FEARFUL="-0.535*E+-0.4375*P+-0.0275*A",
                    HAPPY="0.6175*E+0.3125*P+0.2425*A",
                    SAD="-0.47*E+-0.365*P+-0.3125*A",
                    SURPRISED="0.295*E+0.11*P+0.2375*A")

    vocab = dict(pos=['SUBJECT', 'VERB', 'OBJECT'],
                 verbs=verbs,
                 identities=identities,
                 emotions=emotions)

    spa_vocabs = {}

    spa_vocabs['all'] = spa.Vocabulary(dimensions=D)
    spa_vocabs['epa'] = spa.Vocabulary(dimensions=D)
    spa_vocabs['epa'].parse('E+P+A')
    spa_vocabs['all'].parse('SUBJECT+VERB+OBJECT')

    for category, items in vocab.iteritems():
        spa_vocabs[category] = spa.Vocabulary(dimensions=D)
        if isinstance(items, dict):
            for word, vector in items.iteritems():
                if word not in spa_vocabs[category].keys:
                    spa_vocabs[category].parse(word)

                if word not in spa_vocabs['all'].keys:
                    spa_vocabs['all'].add(word, spa_vocabs[category][word].v)

    return vocab, spa_vocabs
