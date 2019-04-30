
# coding: utf-8

# In[ ]:


#suppress warnings
import warnings
warnings.filterwarnings("ignore")

#import statements
import pandas as pd
import numpy as np
import random
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import glob
import statistics
from pattern.en import suggest
from pycontractions import Contractions
import re
import math
import itertools

#load contractions models
cont = Contractions(api_key="glove-twitter-25")
cont.load_models()

#score two words on similarity (super basic-not amazing)
def similar(w1,w2):
    d = 0
    for i in range(len(w1)):
        d += abs(ord(w1[i])-ord(w2[i]))
    return d

#handle cotractions in a given text
def contract_handle(st,verbose):
    
    t = list(cont.expand_texts([st.replace("’","'")]))[0]
    if verbose > 1:
        print('Text expanded')
    tags = nltk.pos_tag(word_tokenize(str(t)))
    temp = []
    for tag in tags:
        temp.append(tag[0])
            
    if verbose > 1:
        print('Contractions handled')
        
    return ' '.join(temp)

#remove correctly spelled words from a dictionary
def remove_correct(dic):
    okay = ['podcasts','trapped','basilisk','3:23','decorator', 'drapes','bolo','werner', 'herzog','plinth','esoteric',
            'grilled','grams','torey', 'malatia','bartender','sated','incorporeal','...','mundanity','smartphones',
            'invalidating','protesters','satellite-activated','bitcoin','turnip','haters',"'90s",'wonderwall','baseball',
            'shih', 'trenchcoat', 'bulls','millennials','sneezes','motivators','beep', 'jazzed', 'asap','pugs', 'loped',
            'incantation', 'runes','motel','brights',"'s", 'brakes','critiquing','confessing','steeple','adverbs', 
            'elmore','doritos','gourds', 'trombone','intersection','life-changing','vulnerability','panicking', 'podcast',
            'tenuous', 'high-pitched','levitation', 'hiccups', 'suns','simplification','slithering', 'unctuous','phrasing',
            'taco','darkest','um', 'retriever','screenplay']
    missed = []
    for word in dic:
        if ('-' in word) and (len(word) > 1) and ('satellite' not in word):
            subs = []
            for sub in word.split('-'):
                subs.append(suggest(sub)[0])
            if not (('-'.join([i[0] for i in subs]) == word) and (len(set([i[1] for i in subs])) == 1)):
                missed.append((word,dic[word]))
        elif not ((dic[word][0][1] == 1.0) or (word in okay)):
            missed.append((word,dic[word]))
            
    return missed

#correct misspelled words in a given dictionary -- takes a while
def correct(mistakes):
    misses = [i[0] for i in mistakes]
    fixes = [('``','"'),('mamas','mothers'),('kinda','kind of'),('byeee','bye'),("'cuz",'because')]
    for m in misses:
        if m in [i[0] for i in fixes]:
            misses.remove(m)
    #if len(misses) > 0:
     #   print(misses)
    for error in mistakes:
        if (error[1][0][1] >= 0.85) and (error[0] in misses):
            fixes.append((error[0],error[1][0][0]))
            if error[0] in misses:
                misses.remove(error[0])
        elif (error[0] in misses):
            word = error[0]
            alphabet = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b',
                        'n','m']
            for let in alphabet:
                w1 = let+word
                w2 = word+let

                sug_1 = suggest(w1)
                if (sug_1[0][0] == w1) and (sug_1[0][1] >= 0.85):
                    fixes.append((word,w1))
                    if word in misses:
                        misses.remove(word)
                    break;
                    
                sug_2 = suggest(w2)
                if (sug_2[0][0] == w2) and (sug_2[0][1] >= 0.85):
                    fixes.append((word,w2))
                    if word in misses:
                        misses.remove(word)
                    break;
                    
            if word in misses:
                if len(word) <= 5:
                    keywords = [''.join(i) for i in itertools.product(sorted(list(word)), repeat = len(word))]
                    okay = [i for i in keywords if sorted(list(word)) == sorted(list(i))]
                    
                    if len(okay) > 0:
                        poss = []
                        dists = []
                        for test in okay:
                            sug = suggest(test)
                            if sug[0][0] == test:
                                poss.append(sug[0][0])
                                dists.append(similar(word,sug[0][0]))
                        if len(dists) > 0:
                            fixes.append((word,poss[dists.index(min(dists))]))
                            if word in misses:
                                    misses.remove(word)
                        
            if word in misses:
                for i in range(1,len(word)):
                    s = word[0:i]
                    e = word[i:]
                    if not ((len(s) == 1) and (s not in ['a','i']) or (len(e) == 1) and (e not in ['a','i'])):
                        sugs_s = suggest(s)
                        if (sugs_s[0][1] == 1.0) and (sugs_s[0][0] == s):
                            sugs_e = suggest(e)
                            if (sugs_e[0][1] == 1.0) and (sugs_e[0][0] == e):
                                fixes.append((word,sugs_s[0][0]+' '+sugs_e[0][0]))
                                if word in misses:
                                    misses.remove(word)
                                
    #if len(fixes) > 3:
     #   print(fixes)
    return fixes

#spellchecks a given text
def spellcheck(text,verbose):
    d = {}
    
    all_toks = word_tokenize(text)
    
    toks = list(set(all_toks))
    
    if verbose > 1:
        print('Spellcheck tokens created')
    
    for tok in toks:
        if tok not in d:
            d[tok] = suggest(tok)
            
    if verbose > 1:
        print('Spellcheck dictionary created')
        
    missed = remove_correct(d)
    
    if verbose > 1:
        print('Correct words removed')
    
    fixes = correct(missed)
    
    if verbose > 1:
        print('Spellcheck fixes generated')
    
    for fix in fixes:
        #print('Applying fix: ',fix)
        target = fix[0]
        while target in all_toks:
            ind = all_toks.index(fix[0])
            all_toks[ind] = fix[1]
    
    if verbose > 1:
        print('Spellcheck finished')
        
    return ' '.join(all_toks)

#clean a given list of texts
def text_clean(texts,verbose=0):
    
    if verbose > 0:
        print('Starting cleaning...')
        
    #fix contractions
    contractions_fixed = []
    for i in range(len(texts)):
        if verbose > 1:
            print('Contraction handling on text %d of %d...' %(i+1,len(texts)))
        contractions_fixed.append(contract_handle(texts[i].lower().replace('/b',''),verbose))
    
    if verbose > 0:
        print('Contractions fixed...')
        
    #spellcheck
    spellcheck_li = []
    for i in range(len(contractions_fixed)):
        if verbose > 1:
            print('Spellchecking on text %d of %d...' %(i+1,len(contractions_fixed)))
        spellcheck_li.append(spellcheck(contractions_fixed[i],verbose))
        
    if verbose > 0:
        print('Spellcheck complete...')
    
    #return cleaned texts
    return spellcheck_li

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

    while (len(l) < order) and (l[-1] != '+'):
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
            s = s + ' ' + l[i]
            
    return s.strip()

def jacc_calc(generated,texts):
    dists = []
    #take generated string
    w2 = set(generated)
    #get jaccard distance from every proverb
    for text in texts:
        w1 = set(text)
        dists.append(nltk.jaccard_distance(w1,w2))
    
    #calculate mean distance
    d = statistics.mean(dists)
    #return it
    return d

def order_select(good_cutoff,df):
    d = df[df['Percent_Good'] >= good_cutoff]
    if d.empty == True:
        raise ValueError("cutoff is too high...")       
    else:
        return d.loc[d['Jaccard_Distance'].idxmin()]['Order']
    
#give function: texts, how to tokenize, highest order, num to gen, cutoff
#return dictionary of proper order
#how tokenize should be: 'text','sentence'
def gen_optimal_dic(texts,how_tokenize,max_order,reps,cutoff):
    if how_tokenize == 'sentence':
        t = sent_tokenize(' '.join(word_tokenize(' '.join([p.lower() for p in texts]))))
    elif how_tokenize == 'text':
        t = [p.lower() for p in texts]
    else:
        raise ValueError("how_tokenize must be 'sentence' or 'text'...")
        
    orders = range(1,max_order+1)
    
    fulltext = ' '.join(t)
    orde = []
    percs_dup = []
    percs_good = []
    jacc_dist = []
    
    #for each order
    for order in orders:
        orde.append(order)
        #build dict
        dic = dict_gen(t,order)
        sents = []

        #generate a ton of strings
        for j in range(reps):
            #for each generated string
            #add to a list
            sents.append(gen(dic,order))

        #determine how much overlap exists
        perc = sum([s in fulltext for s in sents])/len(sents)

        #calculate percent bad/good
        percs_dup.append(perc)
        percs_good.append(1-perc)

        #remove strings that are bad
        good = [s for s in sents if s not in fulltext]

        #with remaining strings, calculate jaccard distance using method
        dists = [jacc_calc(s,t) for s in good]

        #take mean of all scores
        meandist = statistics.mean(dists)
        jacc_dist.append(meandist)
        
    d = {'Order' : orde, 'Percent_Duplicate' : percs_dup, 'Percent_Good' : percs_good, 'Jaccard_Distance' : jacc_dist}
    df = pd.DataFrame(data=d)
    
    optimal_order = int(order_select(cutoff,df))
    
    good_dict = dict_gen(t,optimal_order)
    
    return good_dict,optimal_order

def make_textlist(texts,tries,length,markov_dic,order):
    
    lens = [len(t) for t in texts]
    q1 = np.percentile(lens, 25)
    q3 = np.percentile(lens, 75)
    
    full_text = ' '.join(sent_tokenize(' '.join(texts).lower()))
    possibles = []
    for i in range(tries):
        possibles.append(gen(markov_dic,order))

    possibles = list(set(possibles))
    possibles = [s for s in possibles if s not in full_text]
    possibles = [s for s in possibles if len(s) >= q1]
    possibles = [s for s in possibles if len(s) <= q3]
    
    dists = []
    for p in possibles:
        dists.append(jacc_calc(p,texts))

    d = {'Sentence' : possibles, 'Distance' : dists}
    df = pd.DataFrame(data=d)
    created_sents = df.sort_values(by='Distance', ascending=True).head(length)['Sentence']
    return created_sents

def markov(texts,length,max_order,how_tokenize='sentence',reps_tries=10000,cutoff=0.75,clean=False):
    t = texts
    
    if clean == True:
        t = text_clean(list(t),verbose=0)
        
    t = [i.lower().replace('\\xe2\\x80\\x93','-').replace('’',"'").replace(',',',').replace('``','') for i in t]
    t = [i.replace('< br/>',' ').replace('\\xe2\\x80\\xa6','...').replace('\\xe2\\x80\\x99',"'") for i in t]
    t = [i.replace('\\xe2\\x80\\x9d','"').replace('\\xe2\\x80\\x9c','"').replace('< /i>',' ') for i in t]
    t = [i.replace('<br/>',' ').replace('<i>',' ').replace('</i>',' ').replace('``','') for i in t]
    
    t = [' '.join(word_tokenize(i)) for i in t]
    print('Texts cleaned...')
    
    okay = False
    count = 0

    while okay != True:
        try:
            dic_i,order = gen_optimal_dic(t,how_tokenize,max_order,reps_tries,cutoff)
            okay = True
        except ValueError as ve:
            print('Error! Message:',ve)
            count += 1
            if count >= 3:
                cutoff = cutoff - 0.05
                count = 0
                
    print('Optimal Markov Model Created...')
    print('Order: %d'%order)
    #print(dic_i)
    
    generated_texts = [t.strip() for t in list(make_textlist(t,reps_tries,length,dic_i,order))]
    
    return generated_texts

def text_to_POS(texts):
    
    t = text_clean(list(texts),verbose=0)
    
    pos = []
    tex = []
    
    for text in t:
        toks = word_tokenize(text)
        tags = nltk.pos_tag(toks)
        pos.append(' '.join([i[1] for i in tags]))
        tex.append(' '.join([i[0] for i in tags]))
        
    return tex,pos

def hidden_markov(texts,length,max_order,how_tokenize='sentence',reps_tries=10000,cutoff=0.75):
    
    t,pos = text_to_POS(texts)
    li = markov(pos,length,max_order,how_tokenize,reps_tries,cutoff,clean=False)
    
    full_text_toks = word_tokenize(' '.join(texts))
    all_pos = ' '.join(pos).split(' ')
    
    pos_to_words_dict = {}
    for i in range(len(all_pos)):
        p = all_pos[i]
        if p not in pos_to_words_dict:
            pos_to_words_dict[p] = [full_text_toks[i]]
        else:
            pos_to_words_dict[p].append(full_text_toks[i])
            
    generated_texts = []
    
    for l in li:
        l = l.replace(' $','$').upper()
        pos_list = l.split(' ')
        t = []
        for p in pos_list:
            t.append(find_next(pos_to_words_dict,p))
        generated_texts.append(' '.join(t))
        
    return generated_texts

