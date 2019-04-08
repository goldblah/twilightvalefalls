
# coding: utf-8

# In[6]:


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


#dict build
def dict_build(lis,order):
    dic = dict()
    
    for l in lis:
        li = word_tokenize(l)
        li.append('+')
        li.insert(0,'*')
        
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


def find_next(self, curr):
    li = self[curr]
    size = len(li)
    ind_next = random.randint(0, size-1)
    return li[ind_next]


def dict_gen(lis,order):
    dic = dict()
    
    for i in range(1,order+1):
        dic.update(dict_build(lis,i))
        
    return dic


def gen(dic, order):
    l = []
    curr = ' *'
    l.append(curr)

    while len(l) < order:
        nex = find_next(dic,curr)
        l.append(nex)
        curr = curr + ' ' + nex
        
    while curr[-1] != '+':
        #get next one
        let = find_next(dic,curr)
        l.append(let)
        #take second letter + next letter, make that curr
        curr = ' '.join(curr.split(' ')[(order*-1):][1:]).strip() + ' ' + let
        if order != 1:
            curr = ' ' + curr
        
    l1toks = ['I','a','1','2','3','4','5','6','7','8','9','0']
    apos = ["'s", "n't"]
    s = ''
    for i in range(1,len(l)-1):
        if ((len(l[i]) == 1) & (l[i] not in l1toks)) or (l[i] in apos):
            s = s+l[i]
        else:
            s = s + ' ' + l[i]
    return s

