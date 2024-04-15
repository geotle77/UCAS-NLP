
import math
from collections import Counter


def computing_entropy(all_character,character_number,times):
     all_likelihood = {}
     for k in all_character.keys():
          all_likelihood[k] = all_character[k]/character_number
     entropy = 0
     for k in all_likelihood.keys():
          single = (all_likelihood[k]) * math.log2(all_likelihood[k])
          entropy = entropy - single
     print("time",times,"text length:",character_number," entropy:",entropy,"\n",end='')
     return [character_number,entropy]