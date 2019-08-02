#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:13:50 2019
# repair data for KES paper

# Purpose: get data from corpus
# | pure NPs | pure NPs + pure VBs | 
# | syntactic NPs | syntactic NPs + cooccurred VBs |

# output: 
l2_frequency(nounlistinFile) | 309,463 extracted terms 
l2_frequency(verblistinFile) | 192,462 extracted terms 

l2_frequency(nounlistinFile2) | 291,488 extracted terms #For all NPS: delete empty, single letter 
l2_frequency(verblistinFile2) | 133,149 extracted terms # For all vbs: clean out the lemmas in stopwords

l2_frequency(NpsPerfileList) | 291,488 extracted NPs #no source data
l2_frequency(VBsPerfileList) | 133,149 extracted VBs #no source data

l2_frequency(NPs_SubjObj_PerfileList) | 113,140 extracted NPs
l2_frequency(VBs_SubjObj_PerfileList) | 116,617 extracted NPs
l2_frequency(VBs_SubjObj_PerfileList2) | 92,338 extracted NPs  # For all vbs: clean out the lemmas in stopwords

l2_frequency(NPsVBs_PerfileList) | 480,394 extracted NPs 
l2_frequency(NPsVBs_SubjObj_PerfileList) | 205,478 extracted NPs
s

@author: zoe
"""

import pandas as pd
import numpy as np 

import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
#from spacy.en import STOP_WORDS #from spacy.lang.en.stop_words import STOP_WORDS | version>2.0
nlp = spacy.load('en_core_web_sm') 
#nlp = spacy.load('en')

from itertools import compress  #to find first ture in boolean

import re
from collections import Counter
from fuzzywuzzy import fuzz

from itertools import chain #to unnest the elements in list
import xlsxwriter




################################## Target data ################################
# pure NPs
# pure NPs+ pure Verbs
# conpelte info


# 1. load data 
file_wos_location = os.getcwd()+'/SourceData/Data.xlsx'
wosData = pd.read_excel(file_wos_location)
wosData_cs = wosData[wosData.Y1==0]   #[6514 rows x 7 columns]

# 2. pure NPs
# from <LDA withOrout hypernym pattern.py> (2,3 mins)

# (2.1) recognize NPs with featurs
nounlistinFile = []
verblistinFile = []
noisy_pos_tags = ['SYM','NUM','PUNCT','SPACE','SCONJ','X']

for file in wosData_cs.Abstract:
    doc = nlp(file)
    
    verbRes = []
    for token in doc:
        if token.pos_ == "VERB":
            verbRes.append((token.text, token.lemma_, token.dep_))
#            print(token.text, token.pos_)
    verblistinFile.append(verbRes)
    
            
    nounRes =[]
    for chunk in doc.noun_chunks:
        i=[]
        j=[]
        indWoNoise=[]
        nounPrase = ""
        
#       2.1 find index of stop words in NP
        for words in chunk:
            i.append(words.lower_ in STOP_WORDS)   #solution3: pefect
            j.append(words.pos_ in noisy_pos_tags)
        indWoNoise=np.bitwise_not(np.bitwise_or(i,j))
        
#       2.2 delete stop words in NP, lemmatized head of NPs
        for i,tokens in enumerate(list(compress(chunk, indWoNoise ))):
            if i==0:
                nounPrase=str(tokens.lemma_)
                
            elif i!=0 and i != len(list(compress(chunk, indWoNoise )))-1:
                if str(tokens)=="'s":
                    nounPrase=nounPrase+str(tokens)
                else:
                    nounPrase=nounPrase+" "+str(tokens)
                    
            elif i!=0 and i == len(list(compress(chunk, indWoNoise )))-1:
                if str(tokens)=="'s":
                    nounPrase=nounPrase+str(tokens)
                else:
                    nounPrase=nounPrase+" "+str(tokens.lemma_)    
        nounRes.append((chunk.text, nounPrase, chunk.root.lemma_, chunk.root.dep_,chunk.root.head.lemma_))  
    nounlistinFile.append(nounRes) 
    
np.save(os.getcwd()+'/intermediateRes/nounlistinFile1', nounlistinFile)
#nounlistinFile = np.load(os.getcwd()+'/intermediateRes/nounlistinFile1'+'.npy')
# l2_frequency(nounlistinFile) | 309,463 extracted terms 
np.save(os.getcwd()+'/intermediateRes/verblistinFile', verblistinFile)
#verblistinFile = np.load(os.getcwd()+'/intermediateRes/verblistinFile'+'.npy')
# l2_frequency(verblistinFile) | 192,462 extracted terms 

 
# (2.2) 
# For all NPS: delete empty, single letter 
nounlistinFile2=[]
for file in nounlistinFile:
    nounRes2=[]
    for item in file:
        if len(item[1]) not in [0,1]:
            nounRes2.append(item)
    nounlistinFile2.append(nounRes2)
np.save(os.getcwd()+'/intermediateRes/nounlistinFile2', nounlistinFile2)
#nounlistinFile2 = np.load(os.getcwd()+'/intermediateRes/nounlistinFile2'+'.npy')
# l2_frequency(nounlistinFile2) | 291,488 extracted terms 

# For all vbs: try to  clean out the lemmas in stopwords
verblistinFile2 = []
for file in verblistinFile:
    verbRes2=[]
    for item in file:
        if item[1] not in STOP_WORDS:
            verbRes2.append(item)
    verblistinFile2.append(verbRes2)
np.save(os.getcwd()+'/intermediateRes/verblistinFile2', verblistinFile2)
#verblistinFile2 = np.load(os.getcwd()+'/intermediateRes/verblistinFile2'+'.npy')
# l2_frequency(verblistinFile2) | 133,149 extracted terms 


    
# (2.3) extract NPs and NPs with syntatic roles | extract VBs and VBs with syntactic roles (clean verbs)
NpsPerfileList = []
#VBsPerfileList = []
NPs_SubjObj_PerfileList = []
VBs_SubjObj_PerfileList = []

for files in nounlistinFile2:
    NpsPerfile = []
#    VBsPerfile = []
    NPs_SubjObj_Perfile = []
    VBs_SubjObj_Perfile = []
    for words in files:
        NpsPerfile.append(words[1])
#        VBsPerfile.append(words[4])
        if words[3] in ['dobj','nsubjpass','nsubj']:  # NP only occurred with specific syntactic roles
            NPs_SubjObj_Perfile.append(words[1]) 
        if words[3] in ['dobj','nsubjpass','nsubj']:  # verbs only occurred with specific syntactic roles
            VBs_SubjObj_Perfile.append(words[4]) 
        
    NpsPerfileList += [NpsPerfile]
#    VBsPerfileList += [VBsPerfile]
    NPs_SubjObj_PerfileList += [NPs_SubjObj_Perfile] 
    VBs_SubjObj_PerfileList += [VBs_SubjObj_Perfile] 

VBsPerfileList = []
for files in verblistinFile2:
    VBsPerfile = []
    for words in files:
        VBsPerfile.append(words[1])
    VBsPerfileList += [VBsPerfile]        
               
        
# clean the pointed VBs (nouns and stopwords are included) | verblistinFile2 | all verbs in files
VBs_SubjObj_PerfileList2 = copy.deepcopy(VBs_SubjObj_PerfileList)
for inx, file in enumerate(VBs_SubjObj_PerfileList):
    intersect = np.intersect1d(file, [i[1] for i in verblistinFile2[inx]]) 
    #only uique terms are stored in "intersect", but we need to store the frequency of terms
    if intersect.size != 0:
        VBs_SubjObj_PerfileList2[inx] = [i for i in file if i in intersect]



# (2.4) merge them into files | all NPs+VBs | syntactic roles NPs+VBs                
NPsVBs_PerfileList = copy.deepcopy(NpsPerfileList)
for idx, files in enumerate(NpsPerfileList):
    NPsVBs_PerfileList[idx].extend(VBsPerfileList[idx])


NPsVBs_SubjObj_PerfileList = copy.deepcopy(NPs_SubjObj_PerfileList)
for idx, files in enumerate(NPs_SubjObj_PerfileList):
    NPsVBs_SubjObj_PerfileList[idx].extend(VBs_SubjObj_PerfileList2[idx])

  

# -------why we cannot use this explicit way (pointer from NPs) to extract verbs-----------
# problem: all verbs cannot be found in the way (NPs --> verbs) 
# reason1: there are half of them is preposition
# reason2: not all verbs are pointed here
# reason3: for all the pointed terms, we can delete prepositions, but it is difficult to delete all nouns (no context, they are prone to be wrong tagged)     
# for example: " rate" and "use" are tagged by Nouns


## for all VBs (and NPs) delete preposition;  delete nouns
## input : STOP_WORDS | VBsPerfileList (if we calculate)
#
## test: whether NPs includes prepositions (answer: no)   
#counter = 0 # the number of files including prepositions
#for files in NpsPerfileList: 
#    test = np.array(set(files)) # # STOP_WORDS is "set", NpsPerfileList is "list"
#    if np.intersect1d(np.array(STOP_WORDS), test).size != 0:
#        counter += 1
#
## purpose: for all VBs (and NPs) delete preposition
#VBsPerfileList2 = copy.deepcopy(VBsPerfileList)
#for inx, files in enumerate(VBsPerfileList): 
#    intersect = np.intersect1d(np.array(list(STOP_WORDS)), np.array(files),return_indices=True) # STOP_WORDS is "set", NpsPerfileList is "list"
#    # result of intersection: |1st: value | 2nd: indice in 1st occurrence array| 3rd: indice in 2nd occurrence array | 
#    # the intersection between "sets" does not work, but "list" works    
#    if intersect[0].size != 0: # intersect.size can not work, because the results are tuples
##       problem with "np.delete"  # np.delete does not work   
##       np.delete(,intersect[2]) #the last occurrence in overlapping is slide [2]
#        VBsPerfileList2[inx] = np.setdiff1d(np.array(VBsPerfileList2[inx]),intersect[0])
## transform from numpy array into list
#VBsPerfileList2 = [list(i) for i in VBsPerfileList2] 
## l2_frequency(VBsPerfileList)- l2_frequency(VBsPerfileList2)  183,144 preposition terms are deleted
##np.save(os.getcwd()+'/intermediateRes/VBsPerfileList2', VBsPerfileList2)
##VBsPerfileList2 = np.load(os.getcwd()+'/intermediateRes/VBsPerfileList2'+'.npy')
## l2_frequency(VBsPerfileList2) | 108,344 extracted NPs
#
#
##purpose: for all VBs (and NPs) delete nouns
#VBsPerfileList3 = copy.deepcopy(VBsPerfileList2)
#for files in VBsPerfileList3[1:2]:
#    for terms in files:
##        print(terms)
#        test = nlp(str(terms))[0]
##        if test.pos_ == "NOUN":
#        print(test.text, test.lemma_, test.pos_, test.tag_, test.dep_,test.shape_, test.is_alpha, test.is_stop)  

np.save(os.getcwd()+'/intermediateRes/NpsPerfileList', NpsPerfileList)
#NpsPerfileList = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')
# l2_frequency(NpsPerfileList) | 291,488 extracted NPs
np.save(os.getcwd()+'/intermediateRes/VBsPerfileList', VBsPerfileList)
#VBsPerfileList = np.load(os.getcwd()+'/intermediateRes/VBsPerfileList'+'.npy')
# l2_frequency(VBsPerfileList) | 133,149 extracted NPs
np.save(os.getcwd()+'/intermediateRes/NPs_SubjObj_PerfileList', NPs_SubjObj_PerfileList)
#NPs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPs_SubjObj_PerfileList) | 113,140 extracted NPs
np.save(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList', VBs_SubjObj_PerfileList)
#VBs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList'+'.npy')
# l2_frequency(VBs_SubjObj_PerfileList) | 113,140 extracted NPs
np.save(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList2', VBs_SubjObj_PerfileList2)
#VBs_SubjObj_PerfileList2 = np.load(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList2'+'.npy')
# l2_frequency(VBs_SubjObj_PerfileList2) | 92,274 extracted NPs
np.save(os.getcwd()+'/intermediateRes/NPsVBs_PerfileList', NPsVBs_PerfileList)
#NPsVBs_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_PerfileList'+'.npy')
# l2_frequency(NPsVBs_PerfileList) | 424,637 extracted NPs 
#TODO: solved: l2_frequency(NPsVBs_PerfileList) = l2_frequency(NpsPerfileList)+l2_frequency(VBsPerfileList)
# 291488 + 133149 = 424637
np.save(os.getcwd()+'/intermediateRes/NPsVBs_SubjObj_PerfileList', NPsVBs_SubjObj_PerfileList)
#NPsVBs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPsVBs_SubjObj_PerfileList) | 205,414 extracted NPs

 