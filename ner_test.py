import nltk
from nltk.tag.stanford import StanfordNERTagger
import pandas as pd
import re
#import pymagic
import sys

jar =  "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/stanford-ner-tagger/stanford-ner.jar"
model = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/stanford-ner-tagger/english.all.3class.distsim.crf.ser.gz"

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

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/gravityfalls/"

file = dir + 'named_entities_gravity_falls.txt'

with open(file, 'r') as myfile:
    ner = myfile.read()

f = open(dir + 'gravity_falls_named_entities.tsv', 'w+')
f.write(ner.replace('::', '\t'))
f.close()

gf_named_entities = []

for n in ner.split('\n'):
    gf_named_entities.append(n.split('::'))










