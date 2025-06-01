import os, sys
import dataloader as dd
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

itokens, otokens = dd.MakeS2SDict('data/pinyin.corpus.examples.txt', dict_file='data/pinyin_word.txt')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())

from rnn_s2s import RNNSeq2Seq

# d_model = 256  
s2s = RNNSeq2Seq(itokens, otokens, 128)

mfile = 'models/pinyin.model.weights.h5'
# lr_scheduler = LRSchedulerPerStep(d_model, 4000) 
# model_saver = ModelCheckpoint(mfile, monitor='ppl', save_best_only=True, save_weights_only=True)

#s2s.model.summary()
opt = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
s2s.compile(opt)

try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

cmds = sys.argv[1:]
if 'train' in cmds:
    # gen = dd.S2SDataGenerator('data/pinyin.corpus.examples.txt', itokens, otokens, batch_size=32, max_len=120)
    X, Y = dd.MakeS2SData('data/pinyin.corpus.examples.txt', itokens, otokens, max_len=120)
    print("Total samples:", len(X))
    print("Validation samples:", int(len(X) * 0.1))
    # rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
    print(X.shape, Y.shape)
    # rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
    # s2s.compile(opt, active_layers=1)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    # s2s.compile(opt, active_layers=2)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    # s2s.compile(opt, active_layers=3)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    # s2s.model.fit(gen, steps_per_epoch=100, epochs=5)
    s2s.model.fit([X,Y], Y[:,1:], batch_size=32, epochs=5, validation_split=0.1)
elif 'test' in cmds:
    print(s2s.decode_sequence(['ji', 'zhi', 'hu', 'die', 'zai', 'yang', 'guang', 'xia', 'fei', 'wu']))
    # print(s2s.decode_sequence('ji','zhi'))
    while True:
        quest = input('> ')
        print(s2s.decode_sequence(quest.split()))
        # rets = s2s.beam_search(quest.split())
        # for x, y in rets: print(x, y)

