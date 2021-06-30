# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:02:57 2020

@author: yuansiyu
"""

import re
import copy
import random
from gensim import corpora,models
import numpy as np
import os
import json
import jieba

def sort_index(doc_ls):
    max_index = 0
    flag = 0
    for ele in doc_ls:
        if ele[1] > flag:
            flag = ele[1]
            max_index = ele[0]
    return max_index
def clean(text):
    filtered_words = []
    text_list = jieba.cut(text, cut_all=False)
    stp2 = ['','t','cn','http','\n']
    for ele in text_list:
        if ele not in stopwords and ele not in stp2 and len(ele) >= 2:
            filtered_words.append(ele)
    return filtered_words

simanan = []

path = 'simanan'
path_list = os.listdir(path)
for filename in path_list:
    test_data_dir = os.path.join(path,filename)
    with open(test_data_dir,'r',encoding = 'utf-8') as load_f: 
        load_dict = json.load(load_f)
        simanan.append(load_dict)

simanan = []
path = 'simanan'
path_list = os.listdir(path)
for filename in path_list:
    test_data_dir = os.path.join(path,filename)
    with open(test_data_dir,'r',encoding = 'utf-8') as load_f: 
        load_dict = json.load(load_f)
        simanan.append(load_dict)

stopwords=[]
for word in open('stopwords.txt','r',encoding = 'utf-8'):
    stopwords.append(word.replace('\n',''))


doc_clean = [clean(data[0]['text']) for data in simanan]
dictionary = corpora.Dictionary(doc_clean)
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
corpus_tfidf = models.TfidfModel(corpus)[corpus]
num_topic = 5

lda = models.LdaModel(corpus_tfidf, num_topics = num_topic,
                  id2word = dictionary, alpha = 'auto',
                  eta = 'auto', minimum_probability = 0.001)


num_show_term = 8
topic_dic = {}
for topic_id in range(num_topic):
    print('\n'+ 'topic ' + str(topic_id) + '\n')
    term_distribute_all = lda.get_topic_terms(topicid = topic_id)
    term_distribute = term_distribute_all[:num_show_term]
    term_distribute = np.array(term_distribute)
    term_id = term_distribute[:,0].astype(np.int)

    ls = []
    for t in term_id:
        ls.append(dictionary.id2token[t])
        print(dictionary.id2token[t])
    topic_dic[topic_id] = ls

num_show_topic = 1
topics = lda.get_document_topics(corpus_tfidf)
print(topics)
topic_index = []
for i in range(len(simanan)):
    topic = np.array(topics[i])
    topic_distribute = np.array(topic[:,1])
    topic_idx = topic_distribute.argsort()[:-num_show_topic-1:-1]
    topic_index.append(topic_idx)
   
doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
index_ls = [0 for i in range(num_topic)]
result = []
for i in range(len(doc_topic)):
    doc = doc_topic[i]
    index = sort_index(doc)
    text = simanan[i][0]['text']
    index_ls[index] = index_ls[index] + 1
    result.append([text, index])

with open('simanan_topic.txt', 'w', encoding='utf-8')as f1:
    for ele in result:
        f1.write(str(ele[1]))
        f1.write('\t')
        f1.write(ele[0])
        f1.write('\n')

f1.close()



