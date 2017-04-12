import numpy as np
import pandas as pd
import os

comp_distance = lambda x, y: np.linalg.norm(x-y, axis=1)
comp_distance_cos = lambda x, y: 1 - \
    np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SenAnalyser(dict):

    def __init__(self, dist=None):
        self.subj, self.verb, self.obj = '', '', ''

        path = os.path.join(os.curdir, 'database')

        self.M = self.load_matrix(path)
        self.vocab, self.emo_vocab = self.load_vocabs(path)

        self.epa_emotions = self.emo_vocab[['e1', 'p1', 'a1']].values

        if dist is None:
            self.dist = comp_distance
        else:
            self.dist = comp_distance_cos

    def load_matrix(self, path):
        # load impression formation matrices
        with np.load(os.path.join(path, 'matrices.npz')) as data:
            # fmat = data['fmat']
            mmat = data['mmat']
        M = mmat
        return M

    def load_vocabs(self, path):
        vocab = pd.read_csv(os.path.join(path, 'epa.txt'), header=None)
        vocab.columns = ['word', 'e1', 'p1', 'a1', 'e2', 'p2', 'a2', 'tmp']
        labels = ['e1', 'p1', 'a1', 'e2', 'p2', 'a2']

        # normalize to [-1,1] range
        vocab[labels] = vocab[labels].apply(lambda x: x/4.)
        vocab[labels] = vocab[labels].clip(-1, None)

        # load list of emotions and create EPA emotion vocabulary
        emo_list = pd.read_csv(
            os.path.join(path, 'basic_emo.txt'), header=None)

        emo_vocab = vocab[vocab.word.isin(emo_list[0].values.tolist())]
        # words = list(vocab.word.values)

        return vocab, emo_vocab

    def get_epa(self, w):
        word_idx = self.vocab.word == w
        row = self.vocab[word_idx].iloc[0]
        _epa = row[['e1', 'p1', 'a1']].values
        return dotdict({'e': _epa[0], 'p': _epa[1], 'a': _epa[2]})

    def get_sentiments(self, subj, verb, obj, emo_display=5):
        self.subj = subj
        self.verb = verb
        self.obj = obj

        s = self.get_epa(subj)
        o = self.get_epa(obj)
        v = self.get_epa(verb)

        t = np.array(
            [1, s.e, s.p, s.a, v.e, v.p, v.a, o.e, o.p, o.a,
                s.e*v.e, s.e*v.p, s.e*v.a, s.p*v.e, s.p*v.p, s.p*o.a, s.a*v.a,
                v.e*o.e, v.e*o.p, v.p*o.e, v.p*o.p, v.p*o.a, v.a*o.e, v.a*o.p,
                s.e*v.e*o.e, s.e*v.p*o.p, s.p*v.p*o.p, s.p*v.p*o.a,
                s.a*v.a*o.a])

        tau = np.dot(self.M, t)

        result_emotions = [[], []]
        for i, (a, b) in enumerate([(0, 3), (6, 9)]):
            distances = self.dist(self.epa_emotions, tau[a:b])

            for j in np.argsort(distances):
                result_emotions[i].append(
                    [self.emo_vocab.iloc[j].word, distances[j]]
                )

        return tau, result_emotions

    def answer_person_emotion(self, person, result):
        if person == self.subj:
            print result[0][:2]
        else:
            print result[1][:2]

    def answer_who_emotion(self, emotion, result):
        n = len(result[0])

        subj_emo, obj_emo = '', ''

        for i in range(n):
            if emotion in result[0][i]:
                subj_emo = result[0][i]
            if emotion in result[1][i]:
                obj_emo = result[1][i]

        th = 1.
        if subj_emo[1] < th or obj_emo[1] < th:
            if obj_emo[1] < th:
                print self.obj, obj_emo
            if subj_emo[1] < th:
                print self.subj, subj_emo
        else:
            print 'No one'
