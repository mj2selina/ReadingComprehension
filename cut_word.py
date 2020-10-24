import jieba
import pandas as pd

def cut_word(sents):
    rtn = {}
    for sent in sents:
        words = jieba.cut(sent)
        for word in words:
            if word in rtn:
                rtn[word] += 1
            else:
                rtn[word] = 1
    return rtn

dureader_str = open('./datasets/dureader.txt').read()
squad_str = open('./datasets/squad.txt').read()
dureader = dureader_str.split('\n')
squad = squad_str.split('\n')

dureader_freq = cut_word(dureader)
squad_freq = cut_word(squad)

dureader_freq_dict = {'words':list(dureader_freq.keys()),'frequency':list(dureader_freq.values())}
squad_freq_dict = {'words':list(squad_freq.keys()),'frequency':list(squad_freq.values())}

dureader_freq_df = pd.DataFrame(dureader_freq_dict)
squad_freq_df = pd.DataFrame(squad_freq_dict)

dureader_freq_df.to_csv('./datasets/dureader_freq.csv')
squad_freq_df.to_csv('./datasets/squad_freq.csv')