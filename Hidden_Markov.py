
# coding: utf-8

# <h2>Suppress Warnings and Import Statements</h2>

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


#import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from bs4 import BeautifulSoup
import time
import html2text
import re
import os
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import glob
from pycontractions import Contractions


# <h2>Create Contractions Handler</h2>

# In[3]:


cont = Contractions(api_key="glove-twitter-25")
cont.load_models()


# <h2>Preprocessing Functions</h2>

# In[5]:


#handles contractions for a given string
def contract_handle(st):
    '''
    Input: a string that may or may not have contractions in it
    Output: a string that has replaced contractions
    '''
    t = list(cont.expand_texts([st.replace("’","'")]))[0]
    tags = nltk.pos_tag(word_tokenize(str(t)))
    temp = []
    
    for tag in tags:
        if tag[1] == 'POS':
            temp.append("is")
        elif tag[1] == 'MD':
            temp.append('will')
        else:
            temp.append(tag[0])
            
    return ' '.join(temp)

#preprocesses a list of texts
def preprocess_text(texts):
    '''
    Input: a list of raw texts
    Output: a list of texts that have had contractions handled and made lower case
    '''
    fin = []
    for t in texts:
        fin.append(contract_handle(t).lower())
        
    return fin

#gets the list of texts converted to POS text
def get_POS_texts(texts):
    '''
    Input: a list of preprocessed texts
    Output: a list of POS strings that correspond to the texts
    '''
    pos = []
    
    for t in texts:
        toks = word_tokenize(t)
        tags_tup = nltk.pos_tag(toks)
        tags = []
        for tag in tags_tup:
            tags.append(tag[1])
            
        tags.append("*END*")
        tags.insert(0,"*START*")
        p = ' '.join(tags)
        pos.append(p)
    
    return pos


# <h2>Dictionary Generation Functions</h2>

# In[6]:


#build a dictionary of a given order for a given list
#dictionary shows given current tokens/substring, what token/word is next
def dict_build(lis,order):
    '''
    Input: a list of texts and an integer order
    Output: a dictionary of given order based on the list of texts
    '''
    dic = dict()
    
    for l in lis:
        li = l.split(' ')
        
        for i in range(len(li) - order):
            tok = ''
            for j in range(order):
                tok = tok + ' ' + li[i+j]
            next_tok = li[i+order]
            #if token not in dict, add it as key, next token in list
            if tok not in dic:
                dic[tok] = [next_tok]
            #else, add next token into list
            else:
                dic[tok].append(next_tok)
            
    return dic

#build a massive dictionary with all orders up to and including the given order
def dict_gen(lis,order):
    '''
    Input: a list of texts and an integer order
    Output: a dictionary of orders range(order+1)
    '''
    dic = dict()
    
    for i in range(1,order+1):
        dic.update(dict_build(lis,i))
        
    return dic

#same as dict_build but builds a dictionary that shows given a POS list the possible corresponding words of given length
def tag_dict_build(texts,pos,sublen):
    '''
    Input: a list of texts and the corresponding POS list and an integer sublength
    Output: a dictionary that shows possible corresponding real words based on the POS substrings
    '''
    sub_dic = dict()
    
    for i in range(len(texts)):
        text = texts[i]
        t = word_tokenize(text)
        t.append("*END*")
        t.insert(0,"*START*")
        
        w = pos[i].split(' ')
        #print(t)
        #print(w)
        for j in range(len(w)-sublen+1):
            if ' '.join(w[j:j+sublen]) not in sub_dic:
                sub_dic[' '.join(w[j:j+sublen])] = [' '.join(t[j:j+sublen])]
            else:
                sub_dic[' '.join(w[j:j+sublen])].append(' '.join(t[j:j+sublen]))
    return sub_dic

#same as dict_gen but builds a dictionary that shows given a POS list the possible corresponding words
def tag_dict_gen(texts,pos,order):
    '''
    Input: a list of texts and the corresponding POS list and an integer sublength
    Output: a dictionary that shows possible corresponding real words based on the POS substrings
    '''
    dic = dict()
    
    for i in range(1,order+1):
        dic.update(tag_dict_build(texts,pos,i))
        
    return dic

#generates next/corresponding token/word given a dictionary and current location
def find_next(self, curr):
    li = self[curr]
    size = len(li)
    ind_next = random.randint(0, size-1)
    return li[ind_next]

#generate text based on given dictionary and order
def gen(dic, order):
    l = []
    curr = ' *START*'
    l.append(curr)

    while len(l) < order:
        nex = find_next(dic,curr)
        l.append(nex)
        curr = curr + ' ' + nex
        
    while l[-1] != '*END*':
        #get next one
        let = find_next(dic,curr)
        l.append(let)
        #take second letter + next letter, make that curr
        curr = ' '.join(curr.split(' ')[(order*-1):][1:]).strip() + ' ' + let
        if order != 1:
            curr = ' ' + curr

    return ' '.join(l)


# <h2>Text Generation</h2>

# In[7]:


#given list of raw text, order, and number of texts to generate
#returns list of generated texts
def text_gen(li,order,num):
    processed = preprocess_text(li)
    pos_texts = get_POS_texts(processed)
    generation_dic = dict_gen(pos_texts,order)
    
    for i in range(num):
        pos_sent = gen(generation_dic,order).strip()
        tag_o = int(len(pos_sent.split(' '))/2)
        td = tag_dict_gen(processed,pos_texts,tag_o)
        st = pos_sent.strip().split(' ')

        fin = []
        for i in range(1,tag_o+1):
            if len(st) % i == 0:
                usable = True
                arr = np.array_split(np.array(st), len(st)/i)
                s = []
                for tok in arr:
                    if ' '.join(tok) in td:
                        s.append(find_next(td,' '.join(tok)))
                    else:
                        usable = False
                if usable == True:
                    fin.append(' '.join(s).split(' ')[1:-1])
        
        sents = []
        for f in fin:
            s = ''
            for i in range(len(f)):
                if i == 0:
                    s += f[i]
                elif (len(f[i]) == 1) & ((f[i] != 'a') and (f[i] != 'i')):
                    s += f[i]
                else:
                    s += ' '+f[i]
            sents.append(s)
    return sents
