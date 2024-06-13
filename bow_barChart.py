import MeCab
import pandas as pd
import codecs
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 30,10


tagger = MeCab.Tagger("mecabrc -u c:/neologd/NEologd.dic")
tagger.parse('')# 実行の準備

tabelog_df = pd.read_csv('./data/tabelog.csv', encoding='ms932', sep=',',skiprows=0)

def mecab_tokenizer(reviews):

    word_list=[]    
    token=tagger.parseToNode(reviews)
    
    while token:
        #print(token.surface,token.feature)   
        hinshi = token.feature.split(',')
        if hinshi[0] =='名詞' and hinshi[1] =='一般' and token.surface != 'うどん' and token.surface !='ラーメン' and token.surface.find('店')== -1:
            word_list.append(token.surface)
        token = token.next
    return word_list







def create_dict(tokens):  # <2>
    # Build vocabulary <3>
    vocabulary = {}

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
            #print(vocabulary)
    return vocabulary

def word_vec(vocabulary,review):
    
    # Build BoW Feature Vector <4>
    word_vector = [0]*len(vocabulary) 
    
    for i, word in enumerate(review):
       
        index = vocabulary[word]
        word_vector[index] += 1

    return word_vector

tokens =[]
for texts in tabelog_df['text']:
    tokens += mecab_tokenizer(texts) 

vocabulary_dic = create_dict(tokens)



bow =[]
for i,row in tabelog_df.iterrows():

    
    reviews = mecab_tokenizer(row['text'])
    word_vector = word_vec(vocabulary_dic,reviews)    
    bow.append(word_vector)

voc_col = [v for v,idx in vocabulary_dic.items()]
bow_df = pd.DataFrame(bow,columns=voc_col)
row_sum = bow_df.sum()
column_name = [w for w,v in row_sum.items() if v >15]

assert len(bow_df)==len(tabelog_df), 'length ummatch'
bow_df = bow_df.loc[:,column_name]
bow_df = pd.concat([tabelog_df,bow_df],axis=1)


with codecs.open("./data/tabelog_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)

#column_name = bow_df.columns[1:]
udon = bow_df[bow_df['ジャンル']=='うどん'].iloc[:,4:]
udon_val = [v for k,v in udon.sum().items()]
ramen = bow_df[bow_df['ジャンル']=='ラーメン'].iloc[:,4:]
ramen_val = [v for k,v in ramen.sum().items()]

fig = plt.figure()
ax1=fig.add_subplot(211,title='うどん棒グラフ')
ax1.bar(np.arange(len(udon_val)), np.array(udon_val), tick_label=column_name, align="center")
ax1.set_xticks(np.arange(len(udon_val)))
ax1.set_xticklabels(column_name, rotation=45, ha='right',fontsize=14)
#ax1.xticks(np.arange(len(udon_val)),udon_key)
ax2=fig.add_subplot(212,title='ラーメン棒グラフ')
ax2.bar(np.arange(len(ramen_val)), np.array(ramen_val), tick_label=column_name, align="center")
ax2.set_xticks(np.arange(len(ramen_val)))
ax2.set_xticklabels(column_name, rotation=45, ha='right',fontsize=14)
#ax1.xticks(np.arange(len(ramen_val)),ramen_key)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()

   