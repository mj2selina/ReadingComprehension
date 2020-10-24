import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import numpy as np
from data_io import dump
from bert_serving.client import BertClient
bc = BertClient()

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class BiDAF:
    def __init__(self, 
                clen, 
                qlen,
                max_char_len,
                vocab_size,           
                cencode,
                qencode,
                embedding_matrix,
                emb_size=50,
                max_features=5000,
                conv_layers=[[10,1],[10,2],[30,3]]):
        """
        双向注意流模型
        :param clen:context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
        """
        self.clen = clen
        self.qlen = qlen
        self.max_features = max_features
        self.emb_size = emb_size
        self.conv_layers = conv_layers
        self.max_char_len = max_char_len
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.cencode = cencode
        self.qencode = qencode
        

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding 
        #cinn = tf.keras.layers.Input(shape=(self.clen,), name='context_input')
        #qinn = tf.keras.layers.Input(shape=(self.qlen,), name='question_input')
        '''
        char_cinn = tf.keras.layers.Input(shape=(self.word_clen,self.max_char_len,),name='char_context_input')
        char_qinn = tf.keras.layers.Input(shape=(self.word_qlen,self.max_char_len,),name='char_question_inputs')
        embedding_layer_char = tf.keras.layers.Embedding(self.max_features,
                                                    self.emb_size,
                                                    embeddings_initializer='uniform',
                                                    )
        cemb = embedding_layer_char(char_cinn)
        qemb = embedding_layer_char(char_qinn)

        c_conv_out = []
        filter_size = sum(list(np.array(self.conv_layers).T[0]))
        assert filter_size == self.emb_size

        for filters,kernel_size in self.conv_layers:
            conv = tf.keras.layers.Conv2D(filters=filters,kernel_size=[kernel_size,self.emb_size],strides=1,activation='relu')
            conv = tf.reduce_max(conv,2)
            c_conv_out.append(conv)

        c_conv_out = tf.keras.layers.concatenate(c_conv_out)

        q_conv_out = []
        for filters,kernel_size in self.conv_layers:
            conv = tf.keras.layers.Conv2D(filters=filters,kernel_size=[kernel_size,self.emb_size],strides=1,activation='relu')
            conv = tf.reduce_max(conv,2)
            q_conv_out.append(conv)
        q_conv_out = tf.keras.layers.concatenate(q_conv_out)
        
        word_cinn = tf.keras.layers.Input(shape=(self.clen,),name='context_input_word')
        word_qinn = tf.keras.layers.Input(shape=(self.qlen,),name='question_input_word')

        embedding_layer_word = tf.keras.layers.Embedding(self.vocab_size,self.emb_size,embeddings_initializer=tf.constant_initializer(self.embedding_matrix),trainable=True)
        emb_cw = embedding_layer_word(word_cinn)
        emb_qw = embedding_layer_word(word_qinn)
        print('emb_cw:',emb_cw.shape)
        cemb = tf.concat([emb_cw,c_conv_out],axis=2)
        qemb = tf.concat([emb_qw,q_conv_out],axis=2)
        print('cemb:',cemb.shape)


        for i in range(self.num_highway_layers):
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question'''

        cencode = self.cencode
        qencode = self.qencode

        # 3.注意流层
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [cinn, qinn]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)


if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './BiDAF_tf2/data/squad/train-v1.1.json',
        './BiDAF_tf2/data/squad/dev-v1.1.json',
        './BiDAF_tf2/data/squad/dev-v1.1.json'
    ])
    train_cc, train_qc, train_cw,train_qw,train_y = ds.get_dataset('./BiDAF_tf2/data/squad/train-v1.1.json')
    test_cc, test_qc,test_cw,test_qw, test_y = ds.get_dataset('./BiDAF_tf2/data/squad/dev-v1.1.json')
    print('原数据的维度：')
    print(train_c.shape, train_q.shape, train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)

    #去重
    train_c_deduplicate, train_q_deduplicate, train_y_deduplicate = ds.deduplicate((train_c, train_q, train_y))
    test_c_deduplicate, test_q_deduplicate, test_y_deduplicate = ds.deduplicate((test_c, test_q, test_y))
    print('去重后数据的维度：')
    print(train_c_deduplicate.shape, train_q_deduplicate.shape, train_y_deduplicate.shape)
    print(test_c_deduplicate.shape, test_q_deduplicate.shape, test_y_deduplicate.shape)

    bc = BertClient(ip='localhost')
    #bert返回的所有向量，包含句向量，cls、seq及padding symbol部分
    cencode_bert_vectors = bc.encode(train_c_deduplicate.extend(test_c_deduplicate),blocking=False)
    qencdoe_bert_vectors = bc.encode(train_q_deduplicate.extend(test_q_deduplicate),blocking=False)

    #获取除句子向量外其他的所有向量
    cencode_bert = [token_vectors[1:] for token_vectors in cencode_bert_vectors ]
    qencdoe_bert = [token_vectors[1:] for token_vectors in qencdoe_bert_vectors]

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        max_char_len=ds.max_char_len,
        vocab_size=len(ds.word_list),
        cencode=cencode_bert,
        qencode=qencdoe_bert,
        embedding_matrix=ds.embedding_matrix,
        emb_size=50,
        max_features=len(ds.charset),
        conv_layers=[[10,1],[10,2],[30,3]]
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_cc, train_qc,train_cw,train_qw], train_y,
        batch_size=64,
        epochs=10,
        validation_data=([test_cc, test_qc,test_cw,test_qw], test_y)
    )
