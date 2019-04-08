import nltk
import pandas as pd
import re
#import pymagic
import sys

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/gravityfalls/"

file = dir + 'A_Tale_of_Two_Stans.txt'

with open(file, 'r') as myfile:
    story = myfile.read().replace('\n\n','\n')

raw_story = r'' + story

matches = re.finditer(r'(([\w ]*\:\s)([\w \.\,\'\-\?\!\(\)\:\"\;]*)\n)', raw_story)

dialogue = []

for m in matches:
    dialogue.append(m[0].replace('\n', ''))

speaker = []
text = []
actions = []

for i,d in enumerate(dialogue):
    actions.append(re.findall(r'\([\w \!\?\:\.\,\-\"\']*\)',d))
    temp_dude = re.sub(r'\([\w \!\?\:\.\,\-]*\)',r'',d)
    temp_dude = temp_dude.replace('\n','').split(': ', 1)
    speaker.append(temp_dude[0])
    text.append(temp_dude[1])

whole_epi = pd.DataFrame(
    {'speaker': speaker,
     'spoken_text': text,
     'actions': actions
    })

print(whole_epi[whole_epi.speaker == 'Ford'])



