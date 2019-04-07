import nltk
import pandas

dir = "/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/gravityfalls/"

file = dir + 'A_Tale_of_Two_Stans.txt'

with open(file, 'r') as myfile:
    story = myfile.read()

print(story)
