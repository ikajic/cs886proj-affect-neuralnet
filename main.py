import sentiment_analyser as sa

inp = raw_input('Input sentence: ')
sentence = inp.split(' ')

analyser = sa.SenAnalyser()

w1, w2, w3 = sentence[0], sentence[1], sentence[2]
tau, results = analyser.get_sentiments(w1, w2, w3)
print tau

print('Available emotions:')
print analyser.emo_vocab.word.values.tolist()

print results

while True:
    inp = raw_input('Question: ')
    if inp == 'q':
        break

    if inp == analyser.subj or inp == analyser.obj:
        analyser.answer_person_emotion(inp, results)
    else:
        analyser.answer_who_emotion(inp, results)

    print 'ok'
else:
    print('Quitting!')
