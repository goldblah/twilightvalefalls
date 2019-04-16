import nltk
from nltk.tag.stanford import StanfordNERTagger
import pandas as pd
import re
#import pymagic
import sys

jar =  "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/stanford-ner-tagger/stanford-ner.jar"
model = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/stanford-ner-tagger/test_gravity_falls_ner.ser.gz"

def epi_df(file):
    with open(file, 'r') as myfile:
        story = myfile.read().replace('\n\n', '\n')

    raw_story = r'' + story

    matches = re.finditer(r'(([\w ]*\:\s)([\w \.\,\'\-\?\!\(\)\:\"\;]*)\n)', raw_story)

    dialogue = []

    for m in matches:
        dialogue.append(m[0].replace('\n', ''))

    speaker = []
    text = []
    actions = []

    for i, d in enumerate(dialogue):
        actions.append(re.findall(r'\([\w \!\?\:\.\,\-\"\']*\)', d))
        temp_dude = re.sub(r'\([\w \!\?\:\.\,\-]*\)', r'', d)
        temp_dude = temp_dude.replace('\n', '').split(': ', 1)
        speaker.append(temp_dude[0])
        text.append(temp_dude[1])

    whole_epi = pd.DataFrame(
        {'speaker': speaker,
         'spoken_text': text,
         'actions': actions
         })

    return whole_epi

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/named_entities_all/"
file = dir + 'A_Tale_of_Two_Stans.txt'

ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

ner_file = dir + 'named_entities_gravity_falls.txt'

with open(ner_file, 'r') as myfile:
    ner = myfile.read()

print(ner)

entities_pieces = []

for n in ner.split('\n'):
    entities_pieces.append(n.split('::'))

final_entities = []

for e in entities_pieces:
    if len(e) > 1 and 'list' not in e[0].lower():
        temp_list = []
        temp_list.append(e[0])
        tags = e[1].split(':')
        for t in tags:
            temp_list.append(t)
        final_entities.append(temp_list)

print(final_entities)


whole_epi = epi_df(file)

print(whole_epi.speaker.unique())

words = nltk.word_tokenize(ner)

f = open(dir + 'gravity_falls_ner.txt', 'w+')
for w in final_entities:
    f.writelines(', '.join(w))
    f.write('\n')
f.close()

#print(ner_tagger.tag(words))











