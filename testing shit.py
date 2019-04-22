import sentiment_analysis_functions
from textblob import TextBlob

with open("/volumes/Hayley's Drive/PycharmProjects/twilightvalefalls/rstories/after_life.text", 'r') as myfile:
    story = myfile.read().replace('\n', ' ')

saf = sentiment_analysis_functions()

story_blob = TextBlob(story)

saf.ngram_analysis(2, story_blob)
