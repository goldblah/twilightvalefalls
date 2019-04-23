class lda:

    import spacy
    spacy.load('en')
    from spacy.lang.en import English
    import sentiment_analysis_functions as saf
    from nltk.corpus import wordnet as wn
    from gensim import corpora
    import pickle
    import gensim
    from nltk.stem.wordnet import WordNetLemmatizer
    import pyLDAvis.gensim

    # def __init__(self):
    #     """
    #     DOOOOOOOOO SOMETHING HERE
    #
    #     """

    def create_corpus(self, name_list, list_epis, ner_file, corpus_dir):

        with open(ner_file, 'r', encoding="utf-8") as myfile:
            ner = myfile.read().split('\n')

        ner_split = list()
        for n in ner:
            ner_split.append(n.split(','))

        cc = self.saf.sentiment_analysis(ner_file)

        for k, i in enumerate(cc.pos_tagger(ner_split, list_epis)):
            story = list()
            for j in i:
                story.append('/'.join(j))

            f = open(corpus_dir + name_list[k] + '.txt', 'w+')
            f.write(' '.join(story))
            f.close()

    def tokenize(self, text):
        lda_tokens = []
        tokens = parser(text)
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

    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def get_lemma2(self, word):
        return WordNetLemmatizer().lemmatize(word)

    def prepare_text_for_lda(self, text):
        tokens = tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in en_stop]
        tokens = [get_lemma(token) for token in tokens]
        return tokens

#     parser = English()
#     nltk.download('stopwords')
#     en_stop = set(nltk.corpus.stopwords.words('english'))
#
# stories_train['tokens'] = [prepare_text_for_lda(srvr), prepare_text_for_lda(the_job),
#                      prepare_text_for_lda(after_life), prepare_text_for_lda(yrg),
#                      prepare_text_for_lda(moon)]
#
# stories_test['tokens'] = [prepare_text_for_lda(mobster), prepare_text_for_lda(memoir)]
#
# dictionary = corpora.Dictionary(stories_train['tokens'])
# corpus = [dictionary.doc2bow(text) for text in stories_train.tokens]
#
# pickle.dump(corpus, open('corpus.pkl', 'wb'))
# dictionary.save('dictionary.gensim')
# print(dictionary)
#
# NUM_TOPICS = 3
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS,
#                                            id2word=dictionary, passes=15)
# ldamodel.save('model3.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# test_bow1 = dictionary.doc2bow(stories_test.tokens[0])
# print(test_bow1)
# print(ldamodel.get_document_topics(test_bow1))
#
# test_bow2 = dictionary.doc2bow(stories_test.tokens[1])
# print(test_bow2)
# print(ldamodel.get_document_topics(test_bow2))
#
# dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
# corpus = pickle.load(open('corpus.pkl', 'rb'))
# lda = gensim.models.ldamodel.LdaModel.load('model3.gensim')
#
# lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display)

import pandas as pd
import os
lda = lda()
dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/"

rstories = pd.read_csv(dir + 'rstories/rs_df.csv', sep='|', index_col=0)

lda.create_corpus(rstories['title'],rstories['handled_text'], dir + 'named_entities_all/rstories_ner.txt', dir+'corpus/rstories/')


