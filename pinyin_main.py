import os, sys
import dataloader as dd
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf 

# tf.config.run_functions_eagerly(True)

itokens, otokens = dd.MakeS2SDict('data/pinyin.corpus.examples.txt', dict_file='data/pinyin_word.txt')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())

from transformer import Transformer, LRSchedulerPerStep

d_model = 256  
s2s = Transformer(itokens, otokens, len_limit=500, d_model=d_model, d_inner_hid=1024, \
                   n_head=4, layers=3, dropout=0.1)

mfile = 'models/pinyin.model.weights.h5'
# lr_scheduler = LRSchedulerPerStep(d_model, 4000) 
# model_saver = ModelCheckpoint(mfile, monitor='ppl', save_best_only=True, save_weights_only=True)

#s2s.model.summary()
opt = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
s2s.compile(opt)

try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

print('Output tokens:', otokens.num())
print('Model output shape:', s2s.model.output.shape)

print("Model weights file exists:", os.path.exists(mfile))
cmds = sys.argv[1:]
if 'train' in cmds:
    # gen = dd.S2SDataGenerator('data/pinyin.corpus.examples.txt', itokens, otokens, batch_size=32, max_len=120)
    # rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
    # rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
    X, Y = dd.MakeS2SData('data/pinyin.corpus.examples.txt', itokens, otokens, max_len=120)

    print("Total samples:", len(X))
    print("Validation samples:", int(len(X) * 0.1))

    opt1 = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    s2s.compile(opt1, active_layers=1)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    s2s.model.fit([X,Y], Y[:,1:], batch_size=32, epochs=5, validation_split=0.1)
    
    opt2 = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    s2s.compile(opt2, active_layers=2)
    s2s.model.fit([X,Y], Y[:,1:], batch_size=32, epochs=5, validation_split=0.1)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    
    opt3 = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    s2s.compile(opt3, active_layers=3)
    s2s.model.fit([X,Y], Y[:,1:], batch_size=32, epochs=5, validation_split=0.1)
    # s2s.model.fit(gen, steps_per_epoch=200, epochs=5, callbacks=[lr_scheduler, model_saver])
    
    if not os.path.exists('models'):
        os.makedirs('models')
    s2s.model.save_weights(mfile)
elif 'test' in cmds:
    s2s.compile(opt, active_layers=3)
    s2s.decode_model = None
    print("target_layer weights shape:", s2s.target_layer.get_weights()[0].shape if s2s.target_layer.get_weights() else "No weights")
    s2s.make_fast_decode_model()
    print("decode model output shape:", s2s.decode_model.output[-1].shape)
    # print(s2s.decode_sequence_fast('ji zhi hu die zai yang guang xia fei wu 。'.split()))
    print(s2s.decode_sequence_fast('ji zhi hu die zai yang guang xia fei wu shi ge zi'.split()))
    while True:
        quest = input('> ')
        print(s2s.decode_sequence_fast(quest.split()))
        rets = s2s.beam_search(quest.split())
        for x, y in rets: print(x, y)



