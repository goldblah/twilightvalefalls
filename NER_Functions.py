
# coding: utf-8

# In[ ]:


#import statements
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

#ner function
#takes in NE list, and text list
def NER_fix(ne_list,text_list):
    fin = []
    for text in text_list:
        tags = nltk.pos_tag(word_tokenize(text))
        for name in ne_list:
            n = name[0].lower()
            if n in text:
                toks = word_tokenize(n)
                for t in tags:
                    i = tags.index(t)
                    if t[0] in toks:
                        indices = []
                        indices.append(i)
                        for k in range(1,len(toks)):
                            if tags[i+k][0] not in toks:
                                break
                            else:
                                indices.append(i+k)
                        if len(indices) == len(toks):
                            indices.reverse()
                            for j in indices:
                                del tags[j]
                            tags.insert(indices[-1],(n,'NNP'))

        fin.append(tags)
        
    return fin

def fix_pos_list(n_entity_li,pos_li):
    names = [n[0].lower() for n in n_entity_li]
    fin = []
    for pos in pos_li:
        p = []
        for tag in pos:
            t = tag
            n = tag[0]
            for ne in names:
                if n == ne.lower():
                    new_tag = n_entity_li[names.index(ne)]
                    t = (ne,'NE'+'::'+new_tag[1]+':'+new_tag[2])
            p.append(t)
            
        fin.append(p)
    return fin


'''
pos_list = NER_fix(ners,texts) 
-- put list of named entities (as tuples (name,category,tagname)) and list of texts in, get back list of tagged texts
-- output list will have named entities be their own tag

fix_pos_list(ners,pos_list)
-- put list of named entities and the tagged texts in, get back the same list of tagged texts, but the named entity tags
are of the form NE::category:tagname (can be altered in line 47)

'''

