class lda:

    import spacy
    spacy.load('en')
    import nltk
    from spacy.lang.en import English
    import sentiment_analysis_functions as saf
    from nltk.corpus import wordnet as wn
    from  nltk.corpus.reader import TaggedCorpusReader
    from gensim import corpora
    import pickle
    import gensim
    from nltk.stem.wordnet import WordNetLemmatizer
    import pyLDAvis.gensim

    def __init__(self, ner_file):
        """
        DOOOOOOOOO SOMETHING HERE

        """
        self.nltk.download('stopwords')
        self.en_stop = set(self.nltk.corpus.stopwords.words('english'))
        self.parser = self.English()
        self.cc = self.saf.sentiment_analysis(ner_file)

        with open(ner_file, 'r', encoding="utf-8") as myfile:
            ner = myfile.read().split('\n')

        self.ner_split = list()
        for n in ner:
            self.ner_split.append(n.split(','))

    def set_corpus(self, fileids, file_path):
        self.corpus = TaggedCorpusReader(file_path, fileids=fileids)


    def create_corpus(self, name_list, list_epis, corpus_dir):

        for k, i in enumerate(self.cc.pos_tagger(self.ner_split, list_epis)):
            story = list()
            for j in i:
                story.append('/'.join(j))

            f = open(corpus_dir + name_list[k].replace(' ','_').replace('/','') + '.txt', 'w+')
            f.write(' '.join(story))
            f.close()

    def __tokenize(self, text):
        lda_tokens = []
        tokens = self.parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    def __get_lemma(self, word):
        lemma = self.wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def __get_lemma2(self, word):
        return self.WordNetLemmatizer().lemmatize(word)

    def prepare_text_for_lda(self, text):
        tokens = self.__tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in self.en_stop]
        tokens = [self.__get_lemma2(token) for token in tokens]
        return tokens

    def hayleys_lda_prep(self, text):
        '''
        Preps a single story for lda analysis
        :param text: a single story list to pass to a tokenizer
        :return: a correctly tokenized story based on the named entity recognition
        '''

        bits = self.cc.pos_tagger(self.ner_split, text)

        story_tokens = []
        for i in bits:
            for j in i:
                story_tokens.append(j[0])

        story_tokens = [token for token in story_tokens if len(token) > 4]
        story_tokens = [token for token in story_tokens if token not in self.en_stop]
        story_tokens = [self.__get_lemma(token) for token in story_tokens]

        print(story_tokens)

        return story_tokens

    def lda(self, num_topics, text):
        tokens = self.hayleys_lda_prep(text)
        dictionary = self.corpora.Dictionary(tokens)

        if self.corpus is None:
            corpus = [dictionary.doc2bow(text) for text in tokens]

        return self.gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)



import pandas as pd
import os
from  nltk.corpus.reader import TaggedCorpusReader

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/"
rstories = pd.read_csv(dir + 'rstories/rs_df.csv', sep='|', index_col=0)
gf = pd.read_csv(dir + 'gravityfalls/gf_eps.csv', sep='|', index_col=0)
wtnv = pd.read_csv(dir + 'wtnv_data/episode_prelim_clean.csv', sep='|', index_col=0)
tz = pd.read_csv(dir + 'twilightzone/tz_df.csv', sep='|', index_col=0)
hhgtg = pd.read_csv(dir + 'hhgtg/hhgtg_df.csv', sep='|', index_col=0)

# filenames = ['named_entities_all/fixed_named_entity_hhgttg.txt', 'named_entities_all/gravity_falls_ner.txt',
#              'named_entities_all/named_entity_tz.txt', 'named_entities_all/wtnv_ner.txt']
# with open(dir + 'named_entities_all/combined_ner.txt', 'w') as outfile:
#     for fname in filenames:
#         with open(dir + fname) as infile:
#             outfile.write(infile.read())

# print('starting r')
# rs_lda = lda(dir + 'named_entities_all/rstories_ner.txt')
# rs_lda.create_corpus(rstories['title'],rstories['handled_text'],  dir+'corpus/rstories/')
# print('finishing r')

# print('starting gf')
# gf_lda = lda(dir + 'named_entities_all/gravity_falls_ner.txt')
# gf_lda.create_corpus(gf['title'], gf['handled_text'], dir+'corpus/gravityfalls/')
# print('finished gf')

# print('starting wtnv')
# wtnv_lda = lda(dir + 'named_entities_all/wtnv_ner.txt')
# wtnv_lda.create_corpus(wtnv['episode_name'], wtnv['text'], dir + 'corpus/wtnv/')
# print('finished wtnv')

# print('starting tz')
# tz_lda = lda(dir + 'named_entities_all/named_entity_tz.txt')
# tz_lda.create_corpus(tz['Title'], tz['Text'], dir + 'corpus/twilightzone/')
# print('finished tzwtnv')

# print('starting hhgtg')
# hhgtg_lda = lda(dir + 'named_entities_all/fixed_named_entity_hhgttg.txt')
# hhgtg_lda.create_corpus(hhgtg['Title'], hhgtg['Text'], dir + 'corpus/hhgtg/')
# print('finished hhgtg')

# corpus = TaggedCorpusReader("/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/corpus/gravityfalls", fileids=gf['title'])
# gf_lda.set_corpus(gf['title'], "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/corpus/gravityfalls")
# gf_lda.hayleys_lda_prep([gf['handled_text'][0]])
