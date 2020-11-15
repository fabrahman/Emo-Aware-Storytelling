"""
Identifying protagonist and their roles during RL training (refer to Appendix A.1 in the paper).
"""

import nltk
from nltk.corpus import names
NAMES = set(names.words())

MALE_NAMES = set(names.words('male.txt'))
FEMALE_NAMES = set(names.words("female.txt"))
FEMALE_NAMES.add('Joeana')


MALE_PEOPLE = {
    "son", "husband", "nephew", "grandpa",
    "granddad", "grampa", "brother",
    "dad", "father", "boyfriend", "boy", "man", "dady", "he"
} | MALE_NAMES

FEMALE_PEOPLE = {
    "daughter", "wife",  "niece",
    "grandma", "sister", "cousin", "grandmom", "momy"
    "mom", "mother", "girlfriend", "nana", "girl", "woman", "she"
} | FEMALE_NAMES

GENERIC_PEOPLE = {"cousin", "friend", "friends", "parents", "children",
                  "grandparents", "fiance", "boss", "manager", 'assistant', 'doctor',
                 'nurse'} #|MALE_PEOPLE|FEMALE_PEOPLE



I_REFS = {"I", "we", "We"} #{"I", "we", "We", "my", "My", "me"}

SOCIAL_GROUPS = {"parents", "children", "grandparents", "family", "friends", "couple", "band", 'kids', 'boys', 'girls'}
SPEAKER = "SPEAKER"

MALE_PRONOUNS = {"his",  "he",  "him"}
FEMALE_PRONOUNS = {"hers",  "she",  "her"}
PRONOUNS = {"my", "our", "theirs", "them", "they"}|MALE_PRONOUNS|FEMALE_PRONOUNS
POSSESIVES = {'my', 'your', 'her', 'his', 'our', 'their'}
ARTICLES = {'a', 'A', 'an', 'An', 'the', 'The'}
ADVERBS = {'when', 'then', 'today', 'one day', 'luckily', 'finally', 'however,', 'so', 'yesterday', 'suddenly', 'one', 'day'}


def check_other_sentences(story):
    track = None
    sentence = story[0]
    tokens = sentence.split()
    
    
    i = 0
    while tokens[i].lower() in ARTICLES or tokens[i].lower() in POSSESIVES or tokens[i] == ',':
        i += 1     
    
    first_tok = tokens[i]
    if first_tok.lower() == 'he':
        track = ('he', 'he')
    elif first_tok.lower() == 'she':  
        track = ('she', 'she')
    elif first_tok.lower() == 'they':  
        track = ('they', 'they')
    return track
    
def find_gender(story):
    track = None
    if len(story) == 0:
        return track
    
    sentence = story[0]
    
    tokens = nltk.word_tokenize(sentence)
    all_tokens = nltk.word_tokenize(" ".join(j for j in story))
    happened_Irefs = I_REFS & set(all_tokens)
    if len(happened_Irefs) >= 1:
        return (list(happened_Irefs)[0], 'I-we')
    
    i = 0
    while tokens[i].lower() in ARTICLES or tokens[i].lower() in POSSESIVES or tokens[i].lower() in ADVERBS or tokens[i] == ',':
        i += 1    
    first_tok = tokens[i].replace('usie', 'Susie').replace('ue', 'Sue')
    if first_tok == 'am':
        first_tok = 'Sam'
    if first_tok in MALE_PEOPLE or first_tok.lower() in MALE_PEOPLE:
        track = (first_tok, 'he')
    elif first_tok in FEMALE_PEOPLE or first_tok.lower() in FEMALE_PEOPLE:
        track = (first_tok, 'she')
    elif first_tok in SOCIAL_GROUPS or first_tok.lower() in SOCIAL_GROUPS:
        track = (first_tok, 'they')
    elif first_tok in GENERIC_PEOPLE or first_tok.lower() in GENERIC_PEOPLE:
        return find_gender(story[1:])
    elif len( set(sentence.split()) & MALE_NAMES) >= 1:
        track = (list(set(sentence.split()) & MALE_NAMES)[0], 'he')
    elif len( set(sentence.split()) & FEMALE_NAMES) >= 1:
        track = (list(set(sentence.split()) & FEMALE_NAMES)[0], 'she')
#     else:
#         track = check_other_sentences(story[1:])
    
    return track


def find_role(text, track=None):
    if not track:
        return 'na'
    if len( I_REFS & set(text)) >= 1:
        return 'x'
    i = 0
    while i < len(text)-1 and (text[i].lower() in ARTICLES or text[i].lower() in POSSESIVES or text[i].lower() in ADVERBS or text[i] == ','):
        i += 1

    if text[i].lower() == track[1] or text[i] == track[0] or track[1] in text: # not exatly rememebr whats the last condition but probably is for sentences that has 2 events and main character appears in the second event
        return 'x'
    return 'na'

def create_pos(story, track):
    pos = ['na'] * len(story) # * 5
    if track != None:
        pos[0] = 'x'
        for i, sent in enumerate(story[1:]):
            tokens= sent.split()
            pos[i+1] = find_role(tokens, track)
    return pos
