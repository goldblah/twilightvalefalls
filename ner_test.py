import nltk
import pandas
import re
#import pymagic
import sys

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/gravityfalls/"

file = dir + 'A_Tale_of_Two_Stans.txt'

with open(file, 'r') as myfile:
    story = myfile.read().replace('\n\n','\n')

raw_story = r'' + story

matches = re.finditer(r'(([\w ]*\:\s)([\w \.\,\'\-\?\!\(\)\:\"\;]*)\n)', raw_story)

print(matches[0][0])




