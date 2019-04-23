class sentiment_analysis():
    import textblob
    import pandas as pd
    import numpy as np
    import nltk
    from textblob.tokenizers import WordTokenizer, SentenceTokenizer

    def __init__(self, ner_file):
        '''
        A helpful little class for us to run everything smoothly in the sentiment analysis portion
        of DSC522 for Dr. Song.
        :return: NONE
        '''

        abbrev = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                  'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                  'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        definition = ['Coordinating conjunction', 'Cardinal number', 'Determiner', 'Existential there', 'Foreign word',
                      'Preposition or subordinating conjunction', 'Adjective', 'Adjective, comparative',
                      'Adjective, superlative', 'List item marker', 'Modal', 'Noun, singular or mass', 'Noun, plural',
                      'Proper noun, singular', 'Proper noun, plural', 'Predeterminer', 'Possessive ending',
                      'Personal pronoun', 'Possessive pronoun', 'Adverb', 'Adverb, comparative', 'Adverb, superlative',
                      'Particle', 'Symbol', 'to', 'Interjection', 'Verb, base form', 'Verb, past tense',
                      'Verb, gerund or present participle', 'Verb, past participle',
                      'Verb, non-3rd person singular present', 'Verb, 3rd person singular present', 'Wh-determiner',
                      'Wh-pronoun', 'Possessive wh-pronoun', 'Wh-adverb']

        self.ppos = self.pd.DataFrame({
            'abbrev': abbrev,
            'definition': definition
        })

        with open(ner_file, 'r', encoding="utf-8") as myfile:
            ner = myfile.read().split('\n')

        self.ner_split = list()
        for n in ner:
            self.ner_split.append(n.split(','))

    def NER_fix(self, ne_list, text_list):
        fin = []
        for text in text_list:
            tags = self.nltk.pos_tag(self.nltk.word_tokenize(text))
            for name in ne_list:
                n = name[0].lower()
                if n in text:
                    toks = self.nltk.word_tokenize(n)
                    for t in tags:
                        i = tags.index(t)
                        if t[0] in toks:
                            indices = []
                            indices.append(i)
                            for k in range(1, len(toks)):
                                if tags[i + k][0] not in toks:
                                    break
                                else:
                                    indices.append(i + k)
                            if len(indices) == len(toks):
                                indices.reverse()
                                for j in indices:
                                    del tags[j]
                                tags.insert(indices[-1], (n, 'NNP'))

            fin.append(tags)

        return fin

    def fix_pos_list(self, n_entity_li, pos_li):
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
                        t = (ne, 'NE' + '::' + new_tag[1].strip() + ':' + new_tag[2].strip())
                p.append(t)

            fin.append(p)
        return fin

    def story_analysis(self, story):
        '''
        story_analysis uses textblob to run basic nlp sentiment analysis on a text file. The
        function returns 3 pandas dataframes containing some basic nlp information on the
        story, broken up by word and sentence. The third dataframe is a list of the noun
        phrases that occur in the story.

        It takes in:
            -a file object: pass a string containing the filename and location
            -an empty dataframe to build the word_info_output table in
            -an empty dataframe to build the sentence_output table in
            -an empty dataframe to build the noun_phrase_output table in
        '''

        wt = self.WordTokenizer()

        story_blob = self.textblob.TextBlob(story.lower(), tokenizer=wt)

        bigrams = self.__ngram_analysis(2, story_blob)

        sentences = self.__sentence_analysis(story_blob.sentences)

        return bigrams, sentences

    def __ngram_analysis(self, n, tbObject):
        en_stop = set(self.nltk.corpus.stopwords.words('english'))
        bigrams = tbObject.ngrams(2)
        all_bigrams = []
        subjectivity = []
        bigrams_pos = []
        polarity = []

        for j in bigrams:
            if not j[0] in en_stop and not j[1] in en_stop:
                all_bigrams.append(j)

        all_bigrams = [' '.join(j) for j in all_bigrams]

        for i in all_bigrams:
            polarity.append(self.textblob.TextBlob(i).polarity)
            subjectivity.append(self.textblob.TextBlob(i).subjectivity)

        tagged_bigrams = self.NER_fix(self.ner_split, all_bigrams)
        tagged_bigrams = self.fix_pos_list(self.ner_split, tagged_bigrams)
        #print(tagged_bigrams)

        for j in tagged_bigrams:
            if not len(j) == 2:
                bigrams_pos.append(list([j[0][1]]))
            else:
                bigrams_pos.append(list([j[0][1],j[1][1]]))

        bigrams_pos = [' '.join(j) for j in bigrams_pos]

        ngram_frame = self.pd.DataFrame({
            'bigram': all_bigrams,
            'pos': bigrams_pos,
            'polarity': polarity,
            'subjectivity': subjectivity
        })

        return(ngram_frame)



    def __sentence_analysis(self, tbObject):
        en_stop = set(self.nltk.corpus.stopwords.words('english'))
        sentences = [str(j) for j in tbObject.sentences]
        subjectivity = []
        sentence_pos = []
        polarity = []

        for s in sentences:
            polarity.append(self.textblob.TextBlob(s).polarity)
            subjectivity.append(self.textblob.TextBlob(s).subjectivity)

        sentence_tags = self.NER_fix(self.ner_split, sentences)
        sentence_tags = self.fix_pos_list(self.ner_split,sentence_tags)

        for i in sentence_tags:
            temp = ''

            for j in range(len(i)):
                temp += i[j][1] + ' '
            sentence_pos.append(temp.strip())

        sentence_frame = self.pd.DataFrame({
            'sentence': sentences,
            'pos': sentence_pos,
            'polarity': polarity,
            'subjectivity': subjectivity
        })

        return (sentence_frame)
