import os
import sys
sys.path.insert(0,'/Users/Evil/learning/开课吧/nlp名企班/项目2_基于大规模预训练模型的机器阅读理解/ReadingComprehension/')
import tensorflow as tf
import numpy as np
np.random.seed(1)

GLOVE_FILE_PATH = './BiDAF_tf2/data/squad/squad_glove.txt'
class Embedding(tf.keras.layers.Layer):
    def __init__(self, 
                vocab_size, 
                emb_size, 
                filters, 
                kernel_size, 
                wrod2id, 
                char2id, 
                glove_path='./BiDAF_tf2/data/squad/squad_glove.txt', 
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.word2id = wrod2id
        self.char2id = char2id
        self.glove_path = glove_path
        self.glove_dict = GloveEmbedding(self.glove_path)
        self.glove_embed = tf.keras.layers.Embedding(self.vocab_size,
                                                    self.emb_size,
                                                    embeddings_initializer=self.glove_dict)
        self.cnn = CharCNNEmbedding(filters=filters,kernel_size=kernel_size,embed_size=emb_size)

    def call(self, x):
        word = self.word2id.get(x)
        char = self.char2id.get(x)
        word_emb = self.glove_embed(word) #(batch_size,seq_len,embed_size)
        char_emb = self.cnn(char)
        concat_emb = tf.concat((word_emb,char_emb),2)
        print('embedding layers:')
        print(concat_emb.shape)
        return concat_emb


class GloveEmbedding(tf.keras.layers.Layer):
    def __init__(self, glove_path='./BiDAF_tf2/data/squad/squad_glove.txt', *args, **kwargs):       
        self.golve_path = glove_path
        self.glove_dict = self.glove_embedding()
        self.glove_dict['UNK'] = [np.random.random() for i in range(50)]
        #self.word2id = word2id
        #self.id2word = id2word
        super().__init__(*args, **kwargs)

    def glove_embedding(self):
        if not os.path.exists(self.golve_path):
            print('not glove vector file exists.')
            return
        squad_glove_dict = {}
        with open(self.golve_path,'r') as f:
            for line in f:
                line = line.split(' ')
                word = line[0]
                vector = line[1:]
                if word in squad_glove_dict:
                    continue
                squad_glove_dict[word] = ' '.join(vector)
        return squad_glove_dict

    def get_glove_embedding(self):
        return self.glove_dict
    
class CharCNNEmbedding(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, embed_size, *args,**kwargs):
        super().__init__(*args,**kwargs)
        #self.emb_size = emb_size
        #self.vocab_size = vocab_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        
    def build(self):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,self.embed_size)
        self.conv = [tf.keras.layers.Conv1D(self.filters[i],self.kernel_size,stride=1,activation='relu',padding='same',input_shape=input_shape[1:]) for i in len(self.filters)]
        self.pooling = tf.nn.max_pool(input,ksize=self.kernel_size,strides=1,padding='same')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self,x):
        # x: (N,seq_len,dims)
        input_shape = x.size()
        word_len = x.size(2)

        x = x.reshape(-1,word_len) #(N*seq_len,word_len)
        x = self.embedding(x) # (N * seq_len,word_len,emb_size)
        x = x.reshape(*input_shape,-1) #(N,seq_len,word_len,emb_size)
        x = tf.reduce_sum(x,1)#按行求和 （N, seq_len,emb_size）

        # CNN
        x = tf.expand_dims(1)#在第二维增加一个维度
        # Conv2d
        #    Input: (N,Cin,Hin,Win)
        #    Output: (N,Cout,Hout,Wout)
        x = [conv(x) for conv in self.conv]
        x = self.pooling(x)
        x = [xx.reshape((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [tf.reduce_sum(xx,2) for xx in x]
        x = tf.concat(x,1)
        x = self.dropout(x)
        return x


'''class CharCNNEmbedding(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self.emb_size = emb_size
        #self.vocab_size = vocab_size
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self):
        #self.embedding = tf.keras.layers.Embedding(self.vocab_size,
        #                                        self.emb_size,
        #                                        embeddings_initializer='uniform',)
        self.conv = [tf.keras.layers.Conv1D(self.filters[i],self.kernel_size,stride=1,activation='relu',padding='same',input_shape=input_shape) for i in len(self.filters)]
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self,x):
        # x: (N,seq_len,dims)
        input_shape = x.size()
        word_len = x.size(2)

        x = x.reshape(-1,word_len) #(N*seq_len,word_len)
        x = self.embedding(x) # (N * seq_len,word_len,emb_size)
        x = x.reshape(*input_shape,-1) #(N,seq_len,word_len,emb_size)
        x = tf.reduce_sum(x,1)#按行求和 （N, seq_len,emb_size）

        # CNN
        x = tf.expand_dims(1)#在第二维增加一个维度
        # Conv2d
        #    Input: (N,Cin,Hin,Win)
        #    Output: (N,Cout,Hout,Wout)
        x = [conv(x) for conv in self.conv]
        x = [xx.reshape((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [tf.reduce_sum(xx,2) for xx in x]
        x = tf.concat(x,1)
        x = self.dropout(x)
        return x'''





    