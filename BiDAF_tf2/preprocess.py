import numpy as np
import data_io as pio
import pandas as pd
import nltk
squad_freq_path = './BiDAF_tf2/data/squad/squad_freq.csv'
GLOVE_FILE_PATH = './BiDAF_tf2/data/squad/squad_glove.txt'
class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.max_char_len = 16
        self.stride = stride
        self.charset = set()
        self.embeddings_index = {}
        self.embeddings_matrix = []
        self.word_list = []
        self.build_charset()
        self.load_glove(GLOVE_FILE_PATH)
        self.build_wordset()
        #self.wordset = set()
        #self.build_squad_wordset()

    def build_squad_wordset(self):
        squad_freq = pd.read_csv(squad_freq_path)
        self.wordset = list(set(squad_freq['words']))
        self.wordset.insert(0,'[UNK]')
        set(self.wordset)
        self.word2id = {self.wordset[i]:i for i in range(len(self.wordset))}
        self.id2word = {value:key for key,value in self.word2id.items()}

    def get_word2id(self):
        return self.word2id
    
    def get_id2word(self):
        return self.id2word

    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)
        
        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']

        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        #print(self.ch2id, self.id2ch)

    def build_wordwet(self):
        idx = list(range(len(self.word_list)))
        self.word2id = dict(zip(self.word_list,idx))
        self.id2word = dict(zip(idx,self.word_listss))

    def dataset_char_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def char_encode(self,context,question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)
        question_encode = self.convert2id_char(max_char_len=self.max_char_len,begin=True,end=True,word_list=q_seg_list)
        print(question_encode)
        left_length = self.max_length - question_encode
        context_encode = self.convert2id_char(max_char_len=self.max_char_len,maxlen=left_length,end=True,word_list=c_seg_list)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length
        return cq_encode

    def word_encode(self,context,question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)
        question_encode = self.convert2id_word(begin=True,end=True,word_list=q_seg_list)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id_word(maxlen=left_length,end=True,word_list=c_seg_list)
        cq_encode = question_encode + context_encode
        assert len(cq_encode) == self.max_length
        return cq_encode

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode
        assert len(cq_encode) == self.max_length
        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def convert2id_char(self,max_char_len=None,maxlen=None,begin=False,end=False,word_list=[]):
        char_list = []
        char_list = [[self.get_id_char('[CLS]')] + [self.get_id_char('[PAD]')] * (max_char_len - 1) * begin + char_list]
        for word in word_list:
            ch = [ch for ch in word]
            if max_char_len is not None:
                ch = ch[:max_char_len]
            ids = list(map(self.get_id_char,ch))
            while len(ids) < max_char_len:
                ids.append(self.get_id_char('[PAD]'))
            char_list.append(np.array(ids))

        if maxlen is not None:
            char_list = char_list[:maxlen - 1 * end]
            char_list += [[self.get_id_char('[PAD]')] * max_char_len] * (maxlen - len(char_list))
        return char_list

    def convert2id_word(self,maxlen=None,begin=False,end=False,word_list=[]):
        ch = [ch for ch in word_list]
        ch = ['cls'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['pad'] * (maxlen - len(ch))
        ids = list(map(self.get_id_word,ch))
        return ids

    def get_id_char(self,ch):
        return self.ch2id.get(ch,self.ch2id['[UNK]'])

    def get_id_word(self,ch):
        return self.word2id(ch,self.word2id['[unk]'])

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_dataset(self, ds_fp):
        ccs, qcs, cws,qws,be = [], [], [],[],[]
        for _, cc, qc, cw, qw, b, e in self.get_data(ds_fp):
            ccs.append(cc)
            qcs.append(qc)
            cws.append(cw)
            qws.append(qw)
            be.append((b, e))
            '''print('**************8')
            print(cs)
            print(len(cs[0]))
            print(qs)
            print(len(qs[0]))
            print(be)
            print(len(be[0]))
            break'''
        return map(np.array, (ccs, qcs, cws,qws, be))

    def deduplicate(self,x):
        cs, qs, be = x
        def deduplicate_item(data):
            dedup = []
            for item in data:
                a = ','.join(map(str,item))
                dedup.append(a)
            dedup = list(set(dedup))
            return dedup
        return map(np.array,(deduplicate_item(cs),deduplicate_item(qs),deduplicate_item(be)))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            '''print('&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print(context)
            print(len(context.split()))
            print(question)
            print(len(question.split()))
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e'''
            c_seg_list = self.seg_text(context)
            q_seg_list = self.seg_text(question)
            c_char_ids = self.get_sent_ids_char(maxlen=self.max_clen,word_list=c_seg_list)
            q_char_ids = self.get_sent_ids_char(maxlen=self.max_qlen,begin=True,word_list=q_seg_list)
            c_word_ids = self.get_sent_ids_word(maxlen=self.max_clen,word_list=c_seg_list)
            q_word_ids = self.get_sent_ids_word(maxlen=self.max_qlen,begin=True,word_list=q_seg_list)
            b,e = answer_start, answer_start + len(text)
            nb = -1
            ne = -1
            len_all_char = 0
            for i,w in enumerate(c_seg_list):
                if i == 0:
                    continue
                if b > len_all_char - 1 and b <= len_all_char + len(w) - 1:
                    b = i + 1
                if e > len_all_char - 1 and e <= len_all_char + len(w) - 1:
                    e = i + 1
                len_all_char += len(w)
            if ne == -1:
                b = e = 0
            yield qid, c_char_ids, q_char_ids, c_word_ids, q_word_ids, b, e

    def get_sent_ids_char(self,maxlen=0,begin=False,end=True,word_list=[]):
        return self.convert2id_char(max_char_len=self.max_char_len,maxlen=maxlen,begin=False,end=True,word_list=word_list)

    def get_sent_ids_word(self,maxlen=0,begin=False,end=True,word_list=[]):
        return self.conver2id_word(maxlen=maxlen,begin=False,end=True,word_list=self.word_list)

    def seg_text(self,text):
        words = [word.lower() for word in nltk.word_tokenize(text)]
        return words

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    def load_glove(self,glove_file_path):
        with open(glove_file_path,encoding='utf-8') as f:
            for line in f:
                word,vector = line.split(maxsplit=1)
                self.embeddings_index[word] = vector
                self.word_list.append(word)
                self.embeddings_matrix.append(vector)
                

if __name__ == '__main__':
    p = Preprocessor([
        './BiDAF_tf2/data/squad/train-v1.1.json',
        './BiDAF_tf2/data/squad/dev-v1.1.json',
    ])
    #print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    #print(p.get_sent_ids('modern stone statue of Mary',20))
    test_c, test_q, test_y = p.get_dataset('./BiDAF_tf2/data/squad/dev-v1.1.json')
    print(len(test_c))
    total = []
    for item in test_c:
        a = ','.join(map(str,item))
        total.append(a)
    total = list(set(total))
    print(len(total))
    with open('./BiDAF_tf2/data/squad/test.txt','w') as f:
        f.write('\n'.join(total))
        
            