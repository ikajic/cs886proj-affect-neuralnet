import numpy as np

import nengo
from nengo import spa

import net_vocabularies as nv
reload(nv)


def t_func(x):
    t = np.array(
        [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
        x[0]*x[3], x[0]*x[4], x[0]*x[5], x[1]*x[3], x[1]*x[4], 
        x[1]*x[8], x[2]*x[5], x[3]*x[6], x[3]*x[7], x[4]*x[6], 
        x[4]*x[7], x[4]*x[8], x[5]*x[6], x[5]*x[7],
        x[0]*x[3]*x[6], x[0]*x[4]*x[7], x[1]*x[4]*x[7], x[1]*x[4]*x[8],
        x[2]*x[5]*x[8]])

    return t


model = nengo.spa.SPA('Syntactic affective reasoning', seed=0)
D = 512
vocab, spa_vocabs = nv.create_spa_vocabularies(D)

with model:
    rng = np.random.RandomState(2)
    
    component_labels = [
        ('identities', 'subject', [0, 3]),
        ('verbs', 'verb', [3, 6]),
        ('identities', 'object', [6, 9])
        ]

    pos_ens = {}
    mappings = {}  # words to EPA
    affect = {}    # EPA to emotion

    with np.load('./database/matrices.npz') as data:
        M = data['mmat']    

    model.sentence = spa.State(dimensions=D, vocab=spa_vocabs['all'])
    
    pre_t = nengo.Ensemble(n_neurons=500, dimensions=9)
    t = nengo.Ensemble(n_neurons=500, dimensions=29)
    tau = nengo.Ensemble(n_neurons=500, dimensions=9)
    
    setattr(model, 'example', 1) 
    
    for vocab_label, pos, (i1, i2) in component_labels:
        # mapping between words and EPA values
        setattr(model, 'map_'+pos, spa.AssociativeMemory(
            input_vocab = spa_vocabs[vocab_label],
            output_vocab = spa_vocabs['epa'],
            input_keys = vocab[vocab_label].keys(),
            output_keys = vocab[vocab_label].values(),
            wta_synapse=0.1
        ))
        mappings[pos] = getattr(model, 'map_'+pos)
        
        # extract pos from input sentence
        nengo.Connection(
            model.sentence.output, mappings[pos].input, 
            transform=spa_vocabs['all'][pos.upper()].get_convolution_matrix().T)
        
        #nengo.Connection(pos_ens[pos].output, mappings[pos].input)
        
        # merge all EPAs in pre_t
        nengo.Connection(
            mappings[pos].output, pre_t[i1:i2], 
            transform=spa_vocabs['epa'].vectors)
        
        if pos in ['subject', 'object']:
            # mapping between EPA values and emotions
            setattr(model, 'affect_'+pos,  spa.AssociativeMemory(
                input_vocab = spa_vocabs['epa'],
                output_vocab = spa_vocabs['emotions'],
                input_keys = vocab['emotions'].values(),
                output_keys = vocab['emotions'].keys(),
                threshold=0.05))
            
            affect[pos] = getattr(model,'affect_'+pos)
            
            nengo.Connection(tau[i1:i2], affect[pos].input,
                transform=2*spa_vocabs['epa'].vectors.T)
    
    with nengo.presets.ThresholdingEnsembles(0.2):
        gate = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(pre_t[0], gate)
    
    nengo.Connection(gate, t[0], transform=3)
    nengo.Connection(pre_t, t[1:], function=t_func)
    
    # multiply it by the impression matrix
    nengo.Connection(t, tau, transform=M)
        
    # person identification
    model.result = spa.State(dimensions=D, vocab=spa_vocabs['pos'])
    model.query = spa.State(dimensions=D, vocab=spa_vocabs['emotions'])
    
    model.comp_subj = spa.Compare(dimensions=D, vocab=spa_vocabs['emotions'])
    nengo.Connection(model.affect_subject.output, model.comp_subj.inputA,
        transform=1)
    nengo.Connection(model.query.output, model.comp_subj.inputB)
    
    model.comp_obj = spa.Compare(dimensions=D, vocab=spa_vocabs['emotions'])
    nengo.Connection(model.affect_object.output, model.comp_obj.inputA,
        transform=1)
    nengo.Connection(model.query.output, model.comp_obj.inputB)
    
    nengo.Connection(model.comp_subj.output, model.result.input,
        transform=spa_vocabs['pos']['SUBJECT'].v[np.newaxis].T)
        
    nengo.Connection(model.comp_obj.output, model.result.input,
        transform=spa_vocabs['pos']['OBJECT'].v[np.newaxis].T)        
    
    sentence1 = 'MOTHER*SUBJECT+HUG*VERB+CHILD*OBJECT'
    sentence2 = 'MOTHER*SUBJECT+ABANDON*VERB+CHILD*OBJECT'
    sentence3 = 'CHILD*SUBJECT+ABANDON*VERB+MOTHER*OBJECT'
    sentence4 = 'BOYFRIEND*SUBJECT+LIE_TO*VERB+GIRLFRIEND*OBJECT'
    sentence5 = 'GIRLFRIEND*SUBJECT+LIE_TO*VERB+BOYFRIEND*OBJECT'
    
    sentence6 = 'BOYFRIEND*SUBJECT + CHEAT*VERB + GIRLFRIEND*OBJECT'
    sentence7 = 'GIRLFRIEND*SUBJECT + CHEAT*VERB + BOYFRIEND*OBJECT'
    
    model.inp1 = spa.Input(sentence=sentence3)
    
    def ask_em(t):
        if 0 < t < 0.2:
            return 'HAPPY'
        if 0.2 < t < 0.4:
            return 'SURPRISED'
        if 0.4 < t < 0.6:
            return 'ANGRY'
        if 0.6 < t < 0.8:
            return 'FEARFUL'
        if 0.8 < t < 1.2:
            return 'SAD'
        else:
            return 'DISGUSTED'

    model.inp2 = spa.Input(query=ask_em)

