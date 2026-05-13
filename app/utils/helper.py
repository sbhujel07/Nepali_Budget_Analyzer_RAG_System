# sentence splitter

import re

#define splitter  func
def split_nepali_sentences(text):
    abbreviations = ['रु', 'डा', 'श्री', 'नं', 'क्र', 'पृ']
    text = text.replace('\n', ' ')
    for abbr in abbreviations:
        text = text.replace(abbr + '.', abbr + '###')
    
    sentences = re.split(r'(?<=[।॥!?])\s+', text)
    sentences = [s.replace('###', '.') for s in sentences]
    sentences = [s.strip() for s in sentences 
                 if s.strip() and not re.fullmatch(r'[०-९]+\.', s.strip())]
    return sentences


