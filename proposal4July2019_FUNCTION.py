#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:50:21 2019

Developed from <proposal4March2019_FUNCTIONS.py>

difference:
  
    
    
    
used functions:
    definedFormat2layer
    replacedByCentralTerms2
    replacedByCentralTerms4 (i did not use it later)
    trainLDAModel
    clusterDistribution4 (new one)
    train_eval_10_v2 (new one) to correct the testing evaluation
    
    
    
@author: ziwei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### library
import pandas as pd
import numpy as np 

import spacy 
#from spacy import STOP_WORDS 
from spacy.lang.en.stop_words import STOP_WORDS #| version>2.0
nlp = spacy.load('en_core_web_sm') 
#nlp = spacy.load('en')

from itertools import compress  #to find first ture in boolean

import re
from collections import Counter
from fuzzywuzzy import fuzz

from itertools import chain #to unnest the elements in list
import xlsxwriter
import copy 

import os




### dataset: 
# | nounlistinFile | nounlistinFile2 | PureNPsPerfileList |
# | PureNPs_VBsPerfileList | 
# | NPOthersInfo2 |NpsOthersPerfileList |

nounlistinFile = np.load('/Users/zoe/anaconda3/LDA experiments/proposal4_Jan2019/nounlistinFile1'+'.npy')
# 297069

# it is not correct way to define nounlistinFile2. It should be simply modified from nounlistinFile.
#nounlistinFile2=[]
#for file in nounlistinFile:
#    nounRes2=[]
#    for item in file:
#        if len(item[1]) not in [0,1] and item[3] in ['dobj','nsubjpass','nsubj']:
#            nounRes2.append(item)
#    nounlistinFile2.append(nounRes2)
##118251
    
PureNPsPerfileList = np.load('/Users/zoe/anaconda3/LDA experiments/proposal4_Jan2019/PureNPsPerfileList'+'.npy')
#len: 118251

PureNPs_VBsPerfileList = np.load('/Users/zoe/anaconda3/LDA experiments/proposal4_Jan2019/PureNPs_VBsPerfileList'+'.npy')
#len: 236502

NPOthersInfo2 = np.load('/Users/zoe/anaconda3/LDA experiments/proposal4_Jan2019/NPOthersFiles2_v2'+'.npy')
# 438257

NpsOthersPerfileList = np.load('/Users/zoe/anaconda3/LDA experiments/proposal4_Jan2019/NpsOthersPerfileList'+'.npy')
# 438257

file_wos_location = '/Users/zoe/anaconda3/database/WebOfScience/Meta-data/Data.xlsx'
wosData = pd.read_excel(file_wos_location)
wosData_cs = wosData[wosData.Y1==0]   #[6514 rows x 7 columns]





NpsPerfileList = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')
# l2_frequency(NpsPerfileList) | 291,488 extracted NPs
VBsPerfileList = np.load(os.getcwd()+'/intermediateRes/VBsPerfileList'+'.npy')
# l2_frequency(VBsPerfileList) | 133,149 extracted NPs
NPs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPs_SubjObj_PerfileList) | 113,140 extracted NPs
VBs_SubjObj_PerfileList2 = np.load(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList2'+'.npy')
# l2_frequency(VBs_SubjObj_PerfileList2) | 92,338 extracted NPs
NPsVBs_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_PerfileList'+'.npy')
# l2_frequency(NPsVBs_PerfileList) | 424,637 extracted NPs 
#TODO: solved: l2_frequency(NPsVBs_PerfileList) = l2_frequency(NpsPerfileList)+l2_frequency(VBsPerfileList) # 291488 + 133149 = 424637
NPsVBs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPsVBs_SubjObj_PerfileList) | 205,478 extracted NPs
















### 
def l2_frequency(l2_list):
    counter = 0
    temp = counter
    for file in l2_list:
        temp = counter
        counter += len(file)
        print(counter -temp)
    return counter

def l2_frequency_files(l2_list):
    counter = 0
    temp = counter
    temp2 = []
    for file in l2_list:
        temp = counter
        counter += len(file)
        temp2.append(counter -temp)
    return temp2

def l3_frequency(l3_list):
    counter = 0
    for file in l3_list:
        temp = counter
        for item in file:
            counter += len(item)
        print(counter-temp)
    return counter

def l3_frequency_u(l3_list):
    counter = 0
    for file in l3_list:
        temp = counter
        for item in file:
            counter += len(np.unique(item))
        print(counter-temp)
    return counter

def l2_plain(l2_list):
    temp = []
    for file in l2_list:
        for item in file:
            temp.append(item)
    return temp

def l3_plain(l3_list):
    temp = []
    for file in l3_list:
        for item in file:
            for i in item:
                temp.append(i)
    return temp
            
def l2_stats(l2_list):
    counter = []
    for file in l2_list:
        counter.append(len(file))
    return average(counter), min(counter), max(counter)










### function
# | definedFormat2layer | 
# | trainLDAModel | trainLDAModel_rdm | 
#  --> filter out words and provide process time
# | replacedByCentralTerms | replacedByCentralTerms2 | replacedByCentralTerms3 (add vebs)| replacedByCentralTerms4
#  --> NPs in files cen be replaced according to wordlist that they give
# | clusterDistribution | clusterDistribution2 | clusterDistribution3 |
#  --> distribution rules from LDA results (only frequent terms are deleted and no
#       rare words are eliminated)
#  --> only give out the reserved terms for first LDA training in <clusterDistribution3>
# | Eval_label_assign |
#  --> label allocation for clustered terms and also evaluation of precision and others
# | num_GS_terms |
#  --> to provide the number of terms in GS over a new corpus after first LDA training



def definedFormat2layer(INPUTLIST_2layer):
    noisy_pos_tags = ['SYM','NUM','PUNCT','SPACE','SCONJ','X']
    
    nounPrase_2=[]
    for file in INPUTLIST_2layer:
        nounPrase_1=[]
        for ele in file:
            doc = nlp(ele)
            
            i=[]
            j=[]
            indWoNoise=[]
            nounPrase = ""
            
            for words in doc:
                i.append(words.lower_ in STOP_WORDS)   #solution3: pefect
                j.append(words.pos_ in noisy_pos_tags)
                indWoNoise=np.bitwise_not(np.bitwise_or(i,j)) 
            
            for i,tokens in enumerate(list(compress(doc, indWoNoise ))):
                if i == 0:
                    nounPrase=str(tokens.lemma_)
    
                elif i!=0 and i != len(list(compress(doc, indWoNoise )))-1:
                    if str(tokens)=="'s":
                        nounPrase=nounPrase+str(tokens.lower_)
                    else:
                        nounPrase=nounPrase+" "+str(tokens.lower_)
    
                elif i!=0 and i == len(list(compress(doc, indWoNoise )))-1:
                    if str(tokens)=="'s":
                        nounPrase=nounPrase+str(tokens.lower_)
                    else:
                        nounPrase=nounPrase+" "+str(tokens.lemma_)              

            nounPrase_1.append(nounPrase)
        nounPrase_2.append(nounPrase_1)
        
    return nounPrase_2




def trainLDAModel(termsPerfileList, filter_no_below, filter_no_above, NUM_topics, KEEP_TOKEN):
    
    import gensim
    
    NPsdictionary = gensim.corpora.Dictionary(termsPerfileList)
    
    NPsdictionary.filter_extremes(no_below=filter_no_below, no_above=filter_no_above, keep_tokens= KEEP_TOKEN) 
    
    bow_corpus = [NPsdictionary.doc2bow(doc) for doc in termsPerfileList]
    
    
    from gensim import  models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    import time
    startTime=time.process_time()
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, num_topics=NUM_topics, id2word=NPsdictionary, chunksize=100, alpha='auto', dtype = np.float64)
    endTime=time.process_time()
    processed_time = endTime-startTime
    
    return lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf



def trainLDAModel_rdm(termsPerfileList, filter_no_below, filter_no_above, NUM_topics, RDM_NM):
    
    import gensim
    
    NPsdictionary = gensim.corpora.Dictionary(termsPerfileList)
    
    NPsdictionary.filter_extremes(no_below=filter_no_below, no_above=filter_no_above)  #len(NPsdictionary) 15058
    
    bow_corpus = [NPsdictionary.doc2bow(doc) for doc in termsPerfileList]
    
    
    from gensim import  models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    import time
    startTime=time.process_time()
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, num_topics=NUM_topics, id2word=NPsdictionary, chunksize=100, alpha='auto',random_state=RDM_NM)
    endTime=time.process_time()
    processed_time = endTime-startTime
    
    return lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf




# input: | PureNPsPerfileList | : pure NPs in file
#        | lables_centralTerms_hyponyms | : central terms in each label file
#        | nounlistinFile2 | : NPs with attributes in file

# output: | hyponym_hypernym_subdomain_list | : <hyponyms, #lables, #indexOfTerms, hyponyms>
#         | Nps_rep_PerfileList | : pure NPs with replaced by hypernyms in file
#         | nounlistinFile2_headRep_v2 | : NPs(replaced) with attributes in file
def replacedByCentralTerms(PureNPsPerfileList, lables_centralTerms_hyponyms, 
                           core_lables3, nounlistinFile2):

    hyponym_hypernym_subdomain_list = []
    Nps_rep_PerfileList = copy.deepcopy(PureNPsPerfileList)
    nounlistinFile2_headRep_v2 = [] # with verb information
    counter = 0
    
    for inx_file, file in enumerate(PureNPsPerfileList):
        print(inx_file)
        nounlistinFile2_headRep_v2_t = []
        hyponym_hypernym_subdomain_list_t = []
        for inx_item, item in enumerate(file):
            searchTerm = item
            term_2 = nounlistinFile2[inx_file][inx_item]
            
            # to test whether <searchTerm> exist in <lables_centralTerms_hyponyms>
            temp = []        
            for inx_i, i in enumerate(lables_centralTerms_hyponyms):
                for inx_j, j in enumerate(i):
                    if searchTerm in j:
                        counter += 1
                        temp.append((searchTerm, inx_i,inx_j, core_lables3[inx_i][inx_j]))
    
            if len(temp)!= 0: 
    
                # to test whether exist more than one label
                temp2 = []
                temp3 = []
                if len(temp)>1:
                    temp2= [len(i[3].split()) for i in temp] #get the length for each option
                    temp3 = temp[np.argmax(temp2)] # find the longest string
                    
                    # store in list
                    if temp3[0][0] != temp3[0][3]:
                        hyponym_hypernym_subdomain_list_t.append(temp3)
                    # np with verb information 
                    nounlistinFile2_headRep_v2_t.append((term_2[0],temp3[3], term_2[2],term_2[3],term_2[4]))
    
                elif len(temp)==1: 
                    
                    # store in list
                    if temp[0][0] != temp[0][3]:
                        hyponym_hypernym_subdomain_list_t.append(temp[0]) 
                    # np with verb information 
                    nounlistinFile2_headRep_v2_t.append((term_2[0],temp[0][3], term_2[2],term_2[3],term_2[4]))                         
            
            elif len(temp)== 0:
                # np with verb information 
                nounlistinFile2_headRep_v2_t.append((term_2[0],term_2[1], term_2[2],term_2[3],term_2[4]))
        
        nounlistinFile2_headRep_v2.append(nounlistinFile2_headRep_v2_t)                        
                
        hyponym_hypernym_subdomain_list.append(hyponym_hypernym_subdomain_list_t)
        
        # for each file, remove original and append new terms 
        if len(hyponym_hypernym_subdomain_list_t) >1 :
        
            for tuples in hyponym_hypernym_subdomain_list_t:
                Nps_rep_PerfileList[inx_file].remove(tuples[0])
                Nps_rep_PerfileList[inx_file].append(tuples[3])
                
        if len(hyponym_hypernym_subdomain_list_t) == 1 :  
            Nps_rep_PerfileList[inx_file].remove(hyponym_hypernym_subdomain_list_t[0][0])
            Nps_rep_PerfileList[inx_file].append(hyponym_hypernym_subdomain_list_t[0][3]) 
        
        print(counter)
        
    return hyponym_hypernym_subdomain_list, Nps_rep_PerfileList, nounlistinFile2_headRep_v2



def replacedByCentralTerms2(NAME_SUBDOMAIN, NEWGOLDNPS, CORE_LABELS):
    
    Nps_rep_PerfileList = copy.deepcopy(NEWGOLDNPS)
    
    temp = []
    for ind_file, file in enumerate(Nps_rep_PerfileList):
        temp2 = []
        for item in file:
            for inx_a, a in enumerate(CORE_LABELS):
                if item in a:
                    temp.append(item)
                    
                    Nps_rep_PerfileList[ind_file].remove(item)
                    temp2.append(NAME_SUBDOMAIN[inx_a])
        
        Nps_rep_PerfileList[ind_file].extend(temp2)
                    
    print(len(temp))
    print(len(np.unique(temp)), np.unique(temp))
    return Nps_rep_PerfileList, temp


# to replace and add some verbs
def replacedByCentralTerms3(NAME_SUBDOMAIN, PURENP, CORE_LABELS, NOUNLISTINFILE, NEWGOLDNPS):
    
    Nps_rep_PerfileList_vb = copy.deepcopy(NEWGOLDNPS)
    Nps_rep_PerfileList_vb_rep = copy.deepcopy(NEWGOLDNPS)
    
    temp = []
    for ind_file, file in enumerate(Nps_rep_PerfileList_vb):
        temp2 =[]
        temp3 =[]
        for item in file:
            for inx_a, a in enumerate(CORE_LABELS):
                if item in a:
                    temp.append(item)
                    # delete original terms
                    Nps_rep_PerfileList_vb_rep[ind_file].remove(item)
                    # append nouns
                    temp2.append(NAME_SUBDOMAIN[inx_a])
                    # append verbs
                    ind_item = PURENP[ind_file].index(item)
                    temp3.append(NOUNLISTINFILE[ind_file][ind_item][4])
                    temp2.append(NOUNLISTINFILE[ind_file][ind_item][4])
           
        Nps_rep_PerfileList_vb_rep[ind_file].extend(temp2)
        Nps_rep_PerfileList_vb[ind_file].extend(temp3)
                                        
    print(len(temp))
    print(len(np.unique(temp)), np.unique(temp))
    
    return Nps_rep_PerfileList_vb, Nps_rep_PerfileList_vb_rep, temp


# the difference from replacedByCentralTerms2(...) 
# add verbs for all NPs extractions

def replacedByCentralTerms4(NAME_SUBDOMAIN, PURENP, CORE_LABELS, NOUNLISTINFILE, NEWGOLDNPS):
    
    Nps_rep_PerfileList_vb = copy.deepcopy(NEWGOLDNPS)
    Nps_rep_PerfileList_vb_rep = copy.deepcopy(NEWGOLDNPS)
    
    temp = []
    for ind_file, file in enumerate(Nps_rep_PerfileList_vb):
        temp2 =[]
        temp3 =[]
        for item in file:
            for inx_a, a in enumerate(CORE_LABELS):
                if item in a:
                    temp.append(item)
                    # delete original terms
                    Nps_rep_PerfileList_vb_rep[ind_file].remove(item)
                    # append nouns
                    temp2.append(NAME_SUBDOMAIN[inx_a])
            # append verbs
            ind_item = PURENP[ind_file].index(item)
            temp3.append(NOUNLISTINFILE[ind_file][ind_item][4])
            temp2.append(NOUNLISTINFILE[ind_file][ind_item][4])
           
        Nps_rep_PerfileList_vb_rep[ind_file].extend(temp2)
        Nps_rep_PerfileList_vb[ind_file].extend(temp3)
                                        
    print(len(temp))
    print(len(np.unique(temp)), np.unique(temp))
    
    return Nps_rep_PerfileList_vb, Nps_rep_PerfileList_vb_rep, temp



def clusterDistribution(bow_corpus, NPsdictionary, lda_model, NameExcel ):
    
    ############## rule 1: reject frequency > 500
    # input: | bow_corpus | NPsdictionary |
    # output:| freq |
    
    # provide frequency of terms in the order of dictionary
    id_freq = []
    for i in bow_corpus:
        for j in i:
           id_freq.append(j) 
    
    freq = np.zeros(len(NPsdictionary), dtype=int)
    for item in id_freq:
        ind = item[0]
        value = item[1]
        freq[ind] = freq[ind] + value
        
    # delete frequent terms
    id_sup500 = np.ix_(freq > 1500) 
    
    rejectedAmount = np.ix_(freq > 1500)[0].size #10 elements  
    print("The amount of frequent terms being rejected: " + str(rejectedAmount))
    
    for k in np.nditer(id_sup500[0]):
        print(NPsdictionary.id2token[int(k)])
    print("----------------------------------------------")
    
    
    ############### rule2: reject "relevance" < 50%
    # calculate relevance (lamba=0) --> topic-term probabilities/ sum of that term
    # input: | lda_model |
    # output:| highest_relev2 | ixTermTps | TermTps |
        
    topic_term_prob = lda_model.get_topics()
    topic_term_relev = np.zeros((len(topic_term_prob),len(NPsdictionary)))
    
    
    for term in range(len(NPsdictionary)):
        term_sum = sum(topic_term_prob[:,[term]])
        for tpc in range(len(topic_term_prob)):
            topic_term_relev[tpc,term] = topic_term_prob[tpc,term]/term_sum
    
    
    highest_relev = []
    for term in range(len(NPsdictionary)):
        highest_relev_value = max(topic_term_relev[:,[term]])
        highest_relev_tpc = np.argmax(topic_term_relev[:,[term]])
        
        if highest_relev_value > 0.5 and max(topic_term_prob[:,term]) > 1e-4:
            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
    
    rejectedAmount = len(NPsdictionary)-len(highest_relev)
    print("The amount of topic-irrelevant terms being rejected: " + str(rejectedAmount)+ '\n')
    
    temp = list(range(len(NPsdictionary)))
    for item in highest_relev:
        temp.remove(item[0])
        
#    for item in temp:
#        print(NPsdictionary.id2token[item], end = ", ")
    print("\n----------------------------------------------")
    
    
    # rule 1 + rule2:
    ind_in_rule1 = [int(k) for k in np.nditer(id_sup500)]
    
    ind_in_rule2 = [k[0] for k in highest_relev]       
    
    ind_in_R1R2=[]
    for item in ind_in_rule2:
        if item not in ind_in_rule1:
            ind_in_R1R2.append(item)
    #len(ind_in_R1R2) 
    
    highest_relev2 = []
    for item in highest_relev:
        if item[0] in ind_in_R1R2:
            highest_relev2.append(item)
    # rank this for better interpretation
    def sortLast(val): 
        return val[2] 
    highest_relev2.sort(key= sortLast, reverse= True)
    
    print("The amount of total residual terms: " + str(len(highest_relev2)))        
            
    
    # form terms into different topics
    ixTermTps=[]
    TermTps=[]
    
    NUM_TPS = topic_term_prob.shape[0]
    
    
    for tpcs in range(NUM_TPS):
        
        TermTps_t = []
        TermTps_w = []
        for item in highest_relev2:
            if item[1] == tpcs: 
                TermTps_t.append(item[0])
                TermTps_w.append(NPsdictionary.id2token[item[0]])
        ixTermTps.append(TermTps_t)
        TermTps.append(TermTps_w)
    
    print("The amount of terms in each topic")
    for ind_line, line in enumerate(TermTps):
        counter = 0 
        for item in line:
            counter += 1 
        print("Topic "+ str(ind_line) + ": " + str(counter))
    
    return ixTermTps, TermTps, highest_relev2


  



def clusterDistribution2(bow_corpus, NPsdictionary, lda_model):
    # "relevance" > 0.2 --> reject 49905 terms (is still too many)
    # "relevance" > 0.001 and max of possibilities > 1e-8 (assume no rules)
    
    ############## rule 1: reject frequency > 1500
    # input: | bow_corpus | NPsdictionary |
    # output:| freq |
    
    # provide frequency of terms in the order of dictionary
    id_freq = []
    for i in bow_corpus:
        for j in i:
           id_freq.append(j) 
    
    freq = np.zeros(len(NPsdictionary), dtype=int)
    for item in id_freq:
        ind = item[0]
        value = item[1]
        freq[ind] = freq[ind] + value
        
    # delete frequent terms
    id_sup500 = [0]
#    id_sup500 = np.ix_(freq > 1500) 
    
    rejectedAmount = np.ix_(freq > 1500)[0].size #10 elements  
    print("----------------------------------------------")
    print("The amount of frequent terms being rejected: " + str(rejectedAmount))
    
    for k in np.nditer(id_sup500[0]):
        print(NPsdictionary.id2token[int(k)])
    print("----------------------------------------------")
    
    
    ############### rule2: reject "relevance" < 50%
    # calculate relevance (lamba=0) --> topic-term probabilities/ sum of that term
    # input: | lda_model |
    # output:| highest_relev2 | ixTermTps | TermTps |
        
    topic_term_prob = lda_model.get_topics()
    topic_term_relev = np.zeros((len(topic_term_prob),len(NPsdictionary)))
    
    
    for term in range(len(NPsdictionary)):
        term_sum = sum(topic_term_prob[:,[term]])
        for tpc in range(len(topic_term_prob)):
            topic_term_relev[tpc,term] = topic_term_prob[tpc,term]/term_sum
    
    
    highest_relev = []
    for term in range(len(NPsdictionary)):
        highest_relev_value = max(topic_term_relev[:,[term]])
        highest_relev_tpc = np.argmax(topic_term_relev[:,[term]])
        
        if highest_relev_value > 0.001 and max(topic_term_prob[:,term]) > 1e-8:
            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
    
    rejectedAmount = len(NPsdictionary)-len(highest_relev)
    print("The amount of topic-irrelevant terms being rejected: " + str(rejectedAmount)+ '\n')
    
    temp = list(range(len(NPsdictionary)))
    for item in highest_relev:
        temp.remove(item[0])
        
#    for item in temp:
#        print(NPsdictionary.id2token[item], end = ", ")
    print("\n----------------------------------------------")
    
    
    # rule 1 + rule2:
    ind_in_rule1 = [int(k) for k in np.nditer(id_sup500)]
    
    ind_in_rule2 = [k[0] for k in highest_relev]       
    
    ind_in_R1R2=[]
    for item in ind_in_rule2:
        if item not in ind_in_rule1:
            ind_in_R1R2.append(item)
    #len(ind_in_R1R2) 
    
    highest_relev2 = []
    for item in highest_relev:
        if item[0] in ind_in_R1R2:
            highest_relev2.append(item)
    # rank this for better interpretation
    def sortLast(val): 
        return val[2] 
    highest_relev2.sort(key= sortLast, reverse= True)
    
    print("The amount of total residual terms: " + str(len(highest_relev2)))        
            
    
    # form terms into different topics
    ixTermTps=[]
    TermTps=[]
    
    NUM_TPS = topic_term_prob.shape[0]
    
    
    for tpcs in range(NUM_TPS):
        
        TermTps_t = []
        TermTps_w = []
        for item in highest_relev2:
            if item[1] == tpcs: 
                TermTps_t.append(item[0])
                TermTps_w.append(NPsdictionary.id2token[item[0]])
        ixTermTps.append(TermTps_t)
        TermTps.append(TermTps_w)
    
    print("The amount of terms in each topic")
    for ind_line, line in enumerate(TermTps):
        counter = 0 
        for item in line:
            counter += 1 
        print("Topic "+ str(ind_line) + ": " + str(counter))
    
    return ixTermTps, TermTps, highest_relev2


# 1st phrase for LDA training (give out the reserved terms only)
def clusterDistribution3(bow_corpus, NPsdictionary, lda_model):
    
       
    ############### rule2: reject "relevance" < 50%
    # calculate relevance (lamba=0) --> topic-term probabilities/ sum of that term
    # input: | lda_model |
    # output:| highest_relev2 | ixTermTps | TermTps |
        
    topic_term_prob = lda_model.get_topics()
    topic_term_relev = np.zeros((len(topic_term_prob),len(NPsdictionary)))
    #Update: 16th April 2019. include all gold NPs for predict #step1: find all terms in gold standard
    termGS = []
    for i in hypernym_1_N_hyponym_human3:
        for j in i:
            for k in j:
              termGS.append(k)  
    
    for term in range(len(NPsdictionary)):
        term_sum = sum(topic_term_prob[:,[term]])
        for tpc in range(len(topic_term_prob)):
            topic_term_relev[tpc,term] = topic_term_prob[tpc,term]/term_sum
    
    
    highest_relev = []
    for term in range(len(NPsdictionary)):
        highest_relev_value = max(topic_term_relev[:,[term]])
        highest_relev_tpc = np.argmax(topic_term_relev[:,[term]])
        
#        if highest_relev_value > 0.5 and max(topic_term_prob[:,term]) > 1e-4:
#            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
#Update: 16th April 2019. include all gold NPs for predict
        if (highest_relev_value > 0.5 and max(topic_term_prob[:,term]) > 1e-4) or (NPsdictionary.id2token[term] in termGS):
            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
    
    rejectedAmount = len(NPsdictionary)-len(highest_relev)
    print("The amount of topic-irrelevant terms being rejected: " + str(rejectedAmount)+ '\n')
    
    temp = list(range(len(NPsdictionary)))
    for item in highest_relev:
        temp.remove(item[0])
    
    reservedTerms = []
    for item in highest_relev:
        reservedTerms.append(NPsdictionary.id2token[item[0]])
    
    return reservedTerms


'''difference with clusterDistribution2:
    reject frequency = half of # of files (here 3000) 
    Add new input variables: | threshold0_freq | threshold1_termProb | threshold2_termSign
    add new output variables: | topic_term_prob | topic_term_relev |
''' 
def clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001):
    # "relevance" > 0.2 --> reject 49905 terms (is still too many)
    # "relevance" > 0.001 and max of possibilities > 1e-8 (assume no rules)
    
    ############## rule 1: reject frequency > 3000
    # input: | bow_corpus | NPsdictionary |
    # output:| freq |
    
    # provide frequency of terms in the order of dictionary
    id_freq = []
    for i in bow_corpus:
        for j in i:
           id_freq.append(j) 
    
    freq = np.zeros(len(NPsdictionary), dtype=int)
    for item in id_freq:
        ind = item[0]
        value = item[1]
        freq[ind] = freq[ind] + value
        
    # delete frequent terms
    
    id_sup_thrld = np.ix_(freq > threshold0_freq) 
    
    rejectedAmount = id_sup_thrld[0].size   
    print("----------------------------------------------")
    print("The amount of frequent terms being rejected: " + str(rejectedAmount))
    print("----------------------------------------------")
    
    
    ############### rule2: reject "relevance" < 50%
    # calculate relevance (lamba=0) --> topic-term probabilities/ sum of that term
    # input: | lda_model |
    # output:| highest_relev2 | ixTermTps | TermTps |
        
    topic_term_prob = lda_model.get_topics()
    topic_term_relev = np.zeros((len(topic_term_prob),len(NPsdictionary)))
    
    
    for term in range(len(NPsdictionary)):
        term_sum = sum(topic_term_prob[:,[term]])
        for tpc in range(len(topic_term_prob)):
            topic_term_relev[tpc,term] = topic_term_prob[tpc,term]/term_sum
    
    
    highest_relev = []
#    threshold1_termProb = 1e-8
#    threshold2_termSign = 0.001
    

    for term in range(len(NPsdictionary)):
        highest_relev_value = max(topic_term_relev[:,[term]])
        highest_relev_tpc = np.argmax(topic_term_relev[:,[term]])
        
        if highest_relev_value > threshold2_termSign and max(topic_term_prob[:,term]) > threshold1_termProb:
            inx_terms_2rules.append(term)
            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
    
    
    rejectedAmount = len(NPsdictionary)-len(highest_relev)
    print("The amount of topic-irrelevant terms being rejected: " + str(rejectedAmount)+ '\n')
    print("----------------------------------------------")
    
#    temp = list(range(len(NPsdictionary)))
#    for item in highest_relev:
#        temp.remove(item[0])      
#    for item in temp:
#        print(NPsdictionary.id2token[item], end = ", ")
#    print("\n----------------------------------------------")
    
    
    # rule 1 + rule2:( get the id of remaining terms)
    if id_sup_thrld[0].size != 0: 
        ind_in_rule1 = [int(k) for k in np.nditer(id_sup_thrld)]
    else:
        ind_in_rule1 = []
    
    ind_in_rule2 = [k[0] for k in highest_relev]       
    
    ind_in_R1R2=[]
    for item in ind_in_rule2:
        if item not in ind_in_rule1:
            ind_in_R1R2.append(item)
    #len(ind_in_R1R2) 

#    # problem: if i used this method, the matrix will shift the location of terms, corresponding to dictionary
#    # the term probability matrix of remaining terms
#    topic_term_prob_2rules = np.delete(topic_term_prob, np.setdiff1d(range(len(NPsdictionary)), ind_in_R1R2), 1) #(50, 71), 1 means column
#    # the term significance matrix of remaining terms
#    topic_term_relev_2rules = np.delete(topic_term_relev, np.setdiff1d(range(len(NPsdictionary)), ind_in_R1R2), 1) 
    print("The amount of total residual terms: " + str(len(ind_in_R1R2)))  
    print("----------------------------------------------")




    # form terms into different topics

    #1).  sort the results by the descending value of probabilities   
    highest_relev2 = []
    for item in highest_relev:
        if item[0] in ind_in_R1R2:
            highest_relev2.append(item)
    # rank this for better interpretation
    def sortLast(val): 
        return val[2] 
    highest_relev2.sort(key= sortLast, reverse= True)
    
           
    #2). aggregate terms by topics 
    ixTermTps=[]  # id of aggregated terms
    TermTps=[]    # aggregated terms
    NUM_TPS = topic_term_prob.shape[0]
    
    for tpcs in range(NUM_TPS):
        
        TermTps_t = []
        TermTps_w = []
        for item in highest_relev2:
            if item[1] == tpcs: 
                TermTps_t.append(item[0])
                TermTps_w.append(NPsdictionary.id2token[item[0]])
        ixTermTps.append(TermTps_t)
        TermTps.append(TermTps_w)
    
    print("The amount of terms in each topic:")
    for ind_line, line in enumerate(TermTps):
        counter = 0 
        for item in line:
            counter += 1 
        print("Topic "+ str(ind_line) + ": " + str(counter))
    
    return ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev













# the function is similar as <clusterDistribution3>, which is to output the reserved terms prepared for the second LDA training
def clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001):
    # "relevance" > 0.2 --> reject 49905 terms (is still too many)
    # "relevance" > 0.001 and max of possibilities > 1e-8 (assume no rules)
    
    ############## rule 1: reject frequency > 3000
    # input: | bow_corpus | NPsdictionary |
    # output:| freq |
    
    # provide frequency of terms in the order of dictionary
    id_freq = []
    for i in bow_corpus:
        for j in i:
           id_freq.append(j) 
    
    freq = np.zeros(len(NPsdictionary), dtype=int)
    for item in id_freq:
        ind = item[0]
        value = item[1]
        freq[ind] = freq[ind] + value
        
    # delete frequent terms
    
    id_sup_thrld = np.ix_(freq > threshold0_freq) 
    
    rejectedAmount = id_sup_thrld[0].size   
    print("----------------------------------------------")
    print("The amount of frequent terms being rejected: " + str(rejectedAmount))
    print("----------------------------------------------")
    
    
    ############### rule2: reject "relevance" < 50%
    # calculate relevance (lamba=0) --> topic-term probabilities/ sum of that term
    # input: | lda_model |
    # output:| highest_relev2 | ixTermTps | TermTps |
        
    topic_term_prob = lda_model.get_topics()
    topic_term_relev = np.zeros((len(topic_term_prob),len(NPsdictionary)))
    
    
    for term in range(len(NPsdictionary)):
        term_sum = sum(topic_term_prob[:,[term]])
        for tpc in range(len(topic_term_prob)):
            topic_term_relev[tpc,term] = topic_term_prob[tpc,term]/term_sum
    
    
    highest_relev = []
#    threshold1_termProb = 1e-8
#    threshold2_termSign = 0.001
    

    for term in range(len(NPsdictionary)):
        highest_relev_value = max(topic_term_relev[:,[term]])
        highest_relev_tpc = np.argmax(topic_term_relev[:,[term]])
        
        if highest_relev_value > threshold2_termSign and max(topic_term_prob[:,term]) > threshold1_termProb:
            inx_terms_2rules.append(term)
            highest_relev.append((term, highest_relev_tpc, float(highest_relev_value)))
    
    
    rejectedAmount = len(NPsdictionary)-len(highest_relev)
    print("The amount of topic-irrelevant terms being rejected: " + str(rejectedAmount)+ '\n')
    print("----------------------------------------------")
    
#    temp = list(range(len(NPsdictionary)))
#    for item in highest_relev:
#        temp.remove(item[0])      
#    for item in temp:
#        print(NPsdictionary.id2token[item], end = ", ")
#    print("\n----------------------------------------------")
    
    
    # rule 1 + rule2:( get the id of remaining terms)
    if id_sup_thrld[0].size != 0: 
        ind_in_rule1 = [int(k) for k in np.nditer(id_sup_thrld)]
    else:
        ind_in_rule1 = []
    
    ind_in_rule2 = [k[0] for k in highest_relev]       
    
    ind_in_R1R2=[]
    for item in ind_in_rule2:
        if item not in ind_in_rule1:
            ind_in_R1R2.append(item)
    #len(ind_in_R1R2) 

#    # problem: if i used this method, the matrix will shift the location of terms, corresponding to dictionary
#    # the term probability matrix of remaining terms
#    topic_term_prob_2rules = np.delete(topic_term_prob, np.setdiff1d(range(len(NPsdictionary)), ind_in_R1R2), 1) #(50, 71), 1 means column
#    # the term significance matrix of remaining terms
#    topic_term_relev_2rules = np.delete(topic_term_relev, np.setdiff1d(range(len(NPsdictionary)), ind_in_R1R2), 1) 
    print("The amount of total residual terms: " + str(len(ind_in_R1R2)))  
    print("----------------------------------------------")




    # form terms into different topics

    #1).  sort the results by the descending value of probabilities   
    highest_relev2 = []
    for item in highest_relev:
        if item[0] in ind_in_R1R2:
            highest_relev2.append(item)
    # rank this for better interpretation
    def sortLast(val): 
        return val[2] 
    highest_relev2.sort(key= sortLast, reverse= True)
    
           
    #2). aggregate terms by topics 
    ixTermTps=[]  # id of aggregated terms
    TermTps=[]    # aggregated terms
    NUM_TPS = topic_term_prob.shape[0]
    
    for tpcs in range(NUM_TPS):
        
        TermTps_t = []
        TermTps_w = []
        for item in highest_relev2:
            if item[1] == tpcs: 
                TermTps_t.append(item[0])
                TermTps_w.append(NPsdictionary.id2token[item[0]])
        ixTermTps.append(TermTps_t)
        TermTps.append(TermTps_w)
    
    print("The amount of terms in each topic:")
    for ind_line, line in enumerate(TermTps):
        counter = 0 
        for item in line:
            counter += 1 
        print("Topic "+ str(ind_line) + ": " + str(counter))
        
    #!!! function to output residual terms
    
    ### error: why it is remove
#    temp = list(range(len(NPsdictionary)))
#    for item in highest_relev2:
#        temp.remove(item[0])
    
    reservedTerms = []
    for item in highest_relev2:
        reservedTerms.append(NPsdictionary.id2token[item[0]])
   
    
    return reservedTerms





def Eval_label_assign(GS_dev, TermTps):

    
    ############ Gold terms extraction and ground-truth label assignment
    counter =0 
    
#    NPswithLabel = []
#    LabelsOfNPs = []
#    for tpc in TermTps:
#        temp = []
#        temp2 = []
#        for item in tpc: 
#            for inx_a, a in enumerate(GS_dev):
#                for b in a:
#                    if item in b:
#                        counter += 1
#                        temp.append(item)
#                        temp2.append(inx_a)                
#        NPswithLabel.append(temp)
#        LabelsOfNPs.append(temp2)
    
    NPswithLabel = []
    LabelsOfNPs = []
    for tpc in TermTps:
        temp = []
        temp2 = []
        for item in tpc: 
            for inx_a, a in enumerate(GS_dev):
                for b in a:
                    if item in b:
                        counter += 1
                        temp.append(item)
                        temp2.append(inx_a)  
                        break
                    
        NPswithLabel.append(temp)
        LabelsOfNPs.append(temp2)

    # counter | 1918 |    
    # number of elements in each topic
    # 16 50 29 13 35 14 20 22 22 21 18 13 22 16 42 7 16 13 22 35 7 43 26 2 24 34 13 17 29 14 31 16 25 17 36 17 33 24 19 2 26 18 847 33 21 22 16 16 30 14
    # some times (zero inside): 25 32 18 13 27 21 27 27 22 28 15 17 14 10 7 17 15 24 28 44 890 16 19 20 12 34 24 25 0 21 28 21 19 18 31 29 13 21 23 28 5 12 4 35 19 34 9 20 34 23 
  

    
    ############### label predication
    # goal: we want to find the most frequent terms to label that cluster it belongs to
    # problem: the labels found by this methods only have 4, rathe than 10.
    # solution: as Fabrice said, as more as possibile for clusters
    
    
    # to change | NPswithLabel| LabelsOfNPs | (delete empty clusters)
    # | LabelsOfNPs | is ground-truth labels 
    # | NPswithLabel | is extracted terms 
    temp = []
    for item in NPswithLabel:
        if item !=[]:
            temp.append(item)
    NPswithLabel = temp
    
    temp = []
    for item in LabelsOfNPs:
        if item !=[]:
            temp.append(item)
    LabelsOfNPs = temp
    
            
    # give the most frequent labels as label of cluster
    LabelsOfClusters = []
    prec_Cluster = []
    for labels in LabelsOfNPs:
        c = Counter(labels)                   
        if c.most_common() != []:
            LabelsOfClusters.append(c.most_common()[0][0])
            prec_Cluster.append(c.most_common()[0][1]/len(labels))
      
    
    # to check whether the assigned labels are completed 
    # np.unique(LabelsOfClusters) # 0,  1,  2,  3,  4,  5,  6,  7, 10 (where 8 and 9 are absent, about <operating system and software engineering)
    
    # initialization of loop <while>
    LabelsOfClusters_uniq = list(np.unique(LabelsOfClusters))
    LabelsOfClass_uniq  = list(range(len(GS_dev)))
    absentLabel = [i for i in LabelsOfClass_uniq if i not in LabelsOfClusters_uniq]
    
    level=0
    while absentLabel != []:
        level += 1 # level==1 --> the second frequent lables are dominant
        for aL in absentLabel:
            for ind_labels, labels in enumerate(LabelsOfNPs):
                c = Counter(labels)                   
                if len(c.most_common()) > level: # make sure that exist
                    if c.most_common()[level][0] == aL:
                        LabelsOfClusters[ind_labels] = aL
                        prec_Cluster[ind_labels] = c.most_common()[level][1]/len(labels) # second place's value 
                        print(ind_labels, aL)
        if level == 10:
            break 
        # statement of judgement in loop <while>
        LabelsOfClusters_uniq = list(np.unique(LabelsOfClusters))
        LabelsOfClass_uniq  = list(range(len(GS_dev)))
        absentLabel = [i for i in LabelsOfClass_uniq if i not in LabelsOfClusters_uniq]
        
    
    # results: 
    # | LabelsOfClusters | :predicated labels of clusters
    # | prec_Cluster | :precision of predicated labels compared to ground truth 
    
    
    
    ### present the assigned labels to each cluster and their terms
    LabelCluster = []
    for inx_tpc, tpc in enumerate(NPswithLabel):
        LabelCluster.append([LabelsOfClusters[inx_tpc]]*len(tpc))
    
    LabelClass = LabelsOfNPs
    
    
    
    ### 2 metrics fro classification and for clustering
    from sklearn.metrics.cluster import adjusted_rand_score
    AdjustedRandScore = adjusted_rand_score(l2_plain(LabelClass), l2_plain(LabelCluster)) # 0.014920520489824758
    
    # average value = 0.34592782782162784 | this times: 0.3214751651768326
    Precision = sum(prec_Cluster)/len(prec_Cluster) 
    
    return NPswithLabel, LabelCluster, LabelClass, AdjustedRandScore, Precision
  

'''
The difference between <Eval_label_assign> and <Eval_label_assign2>:
    #1).  <NPswithLabel> and <LabelsOfNPs>, try to put terms that does not exist in GS as "none"
    #1). (LABLE_PREDIXTION | ) 
    if..case... to distinguish the clusters <label by most freqent terms; 
    label by second freqent terms; without label(all terms are not in GS)> 
    #2-3). (LABLE_PREDIXTION | ) 
    delete them, it does not really make sense to cover more domains, if the id of that domain is weak
    # try to divide the dataset into "terms in GS_train", "terms in GS_test" and "terms of others" 
    where "terms of others"  will be used for prediciton later
'''

def Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary):

    import random
    ############!!! Gold terms extraction and ground-truth label assignment
    
    #1). assign labels to terms in clusters, e.g. NPswithLabel== (TermTps in GS)
    counter =0  #90 / 
    NPswithLabel_all = []
    LabelsOfNPs_all = []
    NPswithLabel_NotTrain = [] 
    
    for tpc in TermTps:
        temp = []
        temp2 = []
        temp3 = []
        for item in tpc: 
            item_in_GS = False # to make the location of item that does not exist in GS
            for inx_a, a in enumerate(GS_train):
                for b in a:
                    if item in b:
                        item_in_GS = True
                        counter += 1
                        temp.append(item)
                        temp2.append(inx_a) 
                        temp3.append(None) 
            if item_in_GS == False:
                        temp.append(None)
                        temp2.append(None)
                        temp3.append(item) #len(NPswithLabel_NotTrain) #20
                        
        NPswithLabel_all.append(temp) #len(NPswithLabel) #50 #l2_frequency(NPswithLabel_all) #110
        LabelsOfNPs_all.append(temp2) #len(LabelsOfNPs) #50 #l2_frequency(LabelsOfNPs_all) #110
        NPswithLabel_NotTrain.append(temp3) #len(NPswithLabel_NotTrain) #50 #l2_frequency(NPswithLabel_NotTrain) #20/110
    
    #2). to generate the list without None values, only location of GS terms is existing 
    NPswithLabel = copy.deepcopy(NPswithLabel_all)
    for i in NPswithLabel:
        while None in i:
            i.remove(None) # because list.remove() only word for first matched items, we change from if to while, for multiple matched items
    #l2_frequency(NPswithLabel) #90
    
    LabelsOfNPs = copy.deepcopy(LabelsOfNPs_all)
    for j in LabelsOfNPs:
        while None in j:
            j.remove(None)  
    #l2_frequency(LabelsOfNPs) #90
    # counter should equal to the number of GS syntactic terms for case1
    # | LabelsOfNPs | is ground-truth labels 
    # | NPswithLabel | is extracted terms 
    
    #3). to delete empty list of NPswithLabel, LabelsOfNPs, e.g. NPswithLabel== (TermTps in GS withput empty lists)
    temp = []
    for item in NPswithLabel_all:
        if item !=[]:
            temp.append(item)
    NPswithLabel_all = temp
    #len(NPswithLabel_all) #30 | #l2_frequency(NPswithLabel_all) #110
    
    temp = []
    temp_inx = []
    # to mark which is NULL cluster 
    for inx_item, item in enumerate(LabelsOfNPs_all):
        if item !=[]:
            temp.append(item)
            temp_inx.append(inx_item)
    LabelsOfNPs_all = temp
    LabelsOfNPs_all_inx = temp_inx
    #len(LabelsOfNPs_all) #30 |  #l2_frequency(LabelsOfNPs_all) #110 

    temp = []
    for item in NPswithLabel_NotTrain:
        if item !=[]:
            temp.append(item)
    NPswithLabel_NotTrain = temp
    #len(NPswithLabel_NotTrain) #30 | #l2_frequency(NPswithLabel_NotTrain) #110
      
    temp = []
    for item in NPswithLabel:
        if item !=[]:
            temp.append(item)
    NPswithLabel = temp
    #len(NPswithLabel) #30 |  #l2_frequency(NPswithLabel) #90 
    
    temp = []
    for item in LabelsOfNPs:
        if item !=[]:
            temp.append(item)
    LabelsOfNPs = temp
    #len(LabelsOfNPs) #30 |  #l2_frequency(LabelsOfNPs) #90 


    ############### label predication
            
    #1). to give the most frequent labels as label of cluster and calculate the correctness for each cluster
    LabelsOfClusters = [] # the voted label for clusters (it can exist None value)
    prec_Cluster = [] # correctness of label predictiton for each cluster
    for labels in LabelsOfNPs_all:
        c = Counter(labels)                   
        if c.most_common() != []:
            
            # if the most frequent terms has label in GS
            if c.most_common()[0][0] != None: 
                LabelsOfClusters.append(c.most_common()[0][0]) #c.most_common()[0][0] first position: value
                prec_Cluster.append(c.most_common()[0][1]/len(labels)) #c.most_common()[0][0] second position: frequency
#            
#            # if the most frequent terms does not have label in GS
#            if c.most_common()[0][0] == None and len(c.most_common())>1:
#                # if the second frequent terms has label in GS
#                if c.most_common()[1] != [] and c.most_common()[1][0] != None:
#                    LabelsOfClusters.append(c.most_common()[1][0]) 
#                    prec_Cluster.append(c.most_common()[1][1]/len(labels)) 
#                # if no terms has label in GS
#                else:
#                    LabelsOfClusters.append(c.most_common()[0][0]) 
#                    prec_Cluster.append(c.most_common()[0][1]/len(labels)) 
                    
#       # randomly error: "LabelCluster.append([LabelsOfClusters[inx_tpc]]*len(tpc))" "IndexError: list index out of range"
#       # problem: the LabelsOfClusters might include "None" as label 
#       # solution: randomly assign a numeral label from exiting labels in list to the position of None value  
#       # to mark the position of random assigned: prec_Cluster == 0 
                    
            
            # if the most frequent terms does not have label in GS
            if c.most_common()[0][0] == None and len(c.most_common())>1:
                # if the second frequent terms has label in GS
                if c.most_common()[1][0] != None:
                    LabelsOfClusters.append(c.most_common()[1][0]) 
                    prec_Cluster.append(c.most_common()[1][1]/len(labels)) 
                # if no terms has label in GS
                else:
                    # if fact, this is none sense. becase it does not exist 2 position with two None in coutner
                    LabelsOfClusters.append(c.most_common()[0][0]) 
                    prec_Cluster.append(c.most_common()[0][1]/len(labels)) 
            elif c.most_common()[0][0] == None and len(c.most_common())==1:
                LabelsOfClusters.append(c.most_common()[0][0]) 
                prec_Cluster.append(c.most_common()[0][1]/len(labels))             
            
     # to delete "None" label and mark their location with O in prec_Cluste

    for inx_i, i in enumerate(LabelsOfClusters):
        if i == None:
            LabelsOfClusters[inx_i] = random.choice(LabelsOfClusters)
            prec_Cluster[inx_i] = 0

            
    #2). to check whether the assigned labels are completed 
#      | purpose of this method|: try to cover the domains as many as possible
#       solution: once we assigned the second frequent labels, we have tried our 
#       best, we do not take efforts to assign more labels that might be weak

#    LabelsOfClusters_uniq = list(np.unique([j for i in LabelsOfNPs for j in i]))
#    LabelsOfClass_uniq  = list(range(len(GS_dev)))
#    absentLabel = [i for i in LabelsOfClass_uniq if i not in LabelsOfClusters_uniq]
#    
#
#    #3). to re-assign the omitted labels for cluster. 
#    # However, it perhaps still exist omitted labels because the original NPs do not cover all of domain names
#    level=0
#    while absentLabel != []:
#        level += 1 # level==1 --> the second frequent lables are dominant
#        for aL in absentLabel:
#            for ind_labels, labels in enumerate(LabelsOfNPs):
#                c = Counter(labels)    # all labels in one cluster               
#                if len(c.most_common()) > level:
#                    if c.most_common()[level][0] == aL:
#                        LabelsOfClusters[ind_labels] = aL
#                        prec_Cluster[ind_labels] = c.most_common()[level][1]/len(labels) # second place's value 
#                        absentLabel.remove(aL) # remove the new generated label from absent list for next loop 
#                        print(ind_labels, aL)
#        if level == 10:
#            break 
    
    
    #3). to present the assigned labels to each cluster and their terms
    # LabelCluster: the new voted labels in a cluster  
    # LabelClass: the real labels in a cluster 
    LabelCluster = []
    for inx_tpc, tpc in enumerate(NPswithLabel_all):
        LabelCluster.append([LabelsOfClusters[inx_tpc]]*len(tpc))
        #l2_frequency(LabelCluster) #110
    LabelClass = LabelsOfNPs_all #l2_frequency(LabelClass) #110
    
    
    # to find the index of terms that are not in GS
    inx_None = []
    for inx_i, i in enumerate(LabelClass):
        for inx_j, j in enumerate(i):
            if j == None:
                inx_None.append((inx_i, inx_j))
                
    # to extract the predicted label vector and GS label vector 
#    LabelClass_train = LabelsOfNPs # l2_frequency(LabelsOfNPs) #90
#    LabelCluster_train = copy.deepcopy(LabelCluster)
#    for inx_i, i in enumerate(LabelCluster_train):
#        for inx_j, j in enumerate(i):
#            for pair in inx_None:
#                if inx_i == pair[0] and inx_j == pair[1]:
#                    LabelCluster_train[inx_i].remove(j)
#                    # l2_frequency(LabelCluster_train) #90
    
    # GS_test                 
    NPswithLabel_test = copy.deepcopy(NPswithLabel_NotTrain)
    NPswithLabel_other = copy.deepcopy(NPswithLabel_NotTrain)
    LabelClass_test = []
    LabelCluster_test = []
    
    for ind_tpcs, tpcs in enumerate(NPswithLabel_NotTrain):
        for ind_nps, nps in enumerate(tpcs):
            if nps != None:
                # to mark that wheter this term belong to test set
                NP_belo_test = False
                for inx_domain, domain in enumerate(GS_test):
                    temp = l2_plain(domain)
                    if nps in temp:
                        NP_belo_test = True
                        LabelClass_test.append(inx_domain)
                        LabelCluster_test.append(LabelsOfClusters[ind_tpcs])

                        # remove testing terms, so that only 'other' terms left
                        NPswithLabel_other[ind_tpcs][ind_nps] = None                
                # remove 'other' terms, so that only 'test' terms left
                if NP_belo_test == False:
                    NPswithLabel_test[ind_tpcs][ind_nps] = None
    
    
    ### prediction
    predict_other = []
    predict_other_Dicinx = [] 
    ixTermTps_wtNone = [i for inx_i, i in enumerate(ixTermTps) if inx_i in LabelsOfNPs_all_inx]
    # NPswithLabel_other | LabelsOfClusters | TermTps | ixTermTps
    for inx_tpcs, tpcs in enumerate(NPswithLabel_other):
        for inx_item, item in enumerate(tpcs):
            if item != None:
                # if <LabelsOfClusters> is squeezed, then re-locate the index of topics 
                if LabelsOfNPs_all_inx != []:
                    if inx_tpcs in LabelsOfNPs_all_inx:
                        inx_temp = LabelsOfNPs_all_inx.index(inx_tpcs)
                        temp = (item, LabelsOfClusters[inx_temp])
                        temp2 = (ixTermTps_wtNone[inx_tpcs][inx_item], LabelsOfClusters[inx_temp])   
                    else:
                        temp = (item, None)
                        temp2 = (ixTermTps_wtNone[inx_tpcs][inx_item], None)   
                else:
                    temp = (item, LabelsOfClusters[inx_tpcs])
                    temp2 = (ixTermTps_wtNone[inx_tpcs][inx_item], LabelsOfClusters[inx_tpcs])   
               
                    predict_other.append(temp)
                    predict_other_Dicinx.append(temp2)
    
    # LabelClass_test | LabelCluster_test | NPswithLabel_test

    res_predict = (len(np.unique(LabelsOfClusters)), predict_other, predict_other_Dicinx)    
    res_nps_test = ([j for i in NPswithLabel_test for j in i if i != None], LabelCluster_test, LabelClass_test)
               
    ### clustering
    from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
    AdjustedRandScore = adjusted_rand_score(LabelClass_test, LabelCluster_test) 
    AdjustedMutualInfoScore = adjusted_mutual_info_score(LabelClass_test, LabelCluster_test) 
    
    # to get the feature space of test terms
    ### topic_term_prob | NPswithLabel_test | NPsdictionary
    NPs_test= [j for i in NPswithLabel_test for j in i if j != None]
    Id_NPs_test = []
    for i in NPs_test:
        Id_NPs_test.append(NPsdictionary.token2id[i])
    feature_NPs_test  = topic_term_prob[:,Id_NPs_test]
    
    SilhouetteScore = silhouette_score(feature_NPs_test.transpose(), LabelCluster_test, metric= 'cosine')
    
    ### classification
    from sklearn.metrics import precision_score, matthews_corrcoef
    # precision == recall == f1-score 
    PrecisionScore = precision_score(LabelClass_test, LabelCluster_test, average= 'micro')
    #Compute the Matthews correlation coefficient (MCC)
    MatthewsCorrcoef = matthews_corrcoef(LabelClass_test, LabelCluster_test)
    
    test_res = (PrecisionScore, MatthewsCorrcoef, AdjustedRandScore, AdjustedMutualInfoScore, SilhouetteScore)
#    train_test_others = (l2_frequency(LabelCluster_train), len(NPs_test), len(NPsdictionary)-l2_frequency(LabelCluster_train)- len(NPs_test))
    # visualization t-SNE
    # from sklearn.manifold import TSNE
    
    
    # res_predict: | number of clusters | tuple(predicted terms,# cluster) | tuple(id of predicted terms,# cluster)
    # res_nps_test: | nps to be tested | predicted cluster for nps| ground-truth label for nps
    
    return res_predict, res_nps_test, test_res





def train_eval_10_v2(TERMSPERFILELIST, FILTER_NO_BELOW, FILTER_NO_ABOVE, NUM_TOPICS,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001):
    
    GS_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy') 
    GS_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy') 
    PrecisionScore = [] 
    MatthewsCorrcoef = [] 
    AdjustedRandScore = [] 
    AdjustedMutualInfoScore = [] 
    SilhouetteScore = []

    
    for i in range(10):


        lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=NUM_TOPICS, KEEP_TOKEN = keepToken)         
        ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

        res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)
        
        PrecisionScore.append(test_res[0])
        MatthewsCorrcoef.append(test_res[1])
        AdjustedRandScore.append(test_res[2])
        AdjustedMutualInfoScore.append(test_res[3])
        SilhouetteScore.append(test_res[4])

        
    avg_PrecisionScore = sum(PrecisionScore)/len(PrecisionScore)
    avg_MatthewsCorrcoef = sum(MatthewsCorrcoef)/len(MatthewsCorrcoef)
    avg_AdjustedRandScore = sum(AdjustedRandScore)/len(AdjustedRandScore)
    avg_AdjustedMutualInfoScore = sum(AdjustedMutualInfoScore)/len(AdjustedMutualInfoScore)
    avg_SilhouetteScore = sum(SilhouetteScore)/len(SilhouetteScore)
    
    res_avg = (avg_PrecisionScore,avg_MatthewsCorrcoef,avg_AdjustedRandScore,avg_AdjustedMutualInfoScore,avg_SilhouetteScore)
    res_stas = (PrecisionScore,MatthewsCorrcoef,AdjustedRandScore,AdjustedMutualInfoScore,SilhouetteScore)
    
    return res_avg,res_stas






def train_eval_10(TERMSPERFILELIST, FILTER_NO_BELOW, FILTER_NO_ABOVE, NUM_TOPICS):

    AdjustedRandScore_t=[]
    Precision_t=[]
    for i in range(10):
        lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(TERMSPERFILELIST, FILTER_NO_BELOW, FILTER_NO_ABOVE,NUM_TOPICS) 
        ixTermTps, TermTps, highest_relev2 = clusterDistribution2(bow_corpus, NPsdictionary, lda_model)
        
        NPswithLabel, LabelCluster, LabelClass, AdjustedRandScore, Precision = Eval_label_assign(hypernym_1_N_hyponym_human3, TermTps)
        AdjustedRandScore_t.extend([AdjustedRandScore])
        Precision_t.extend([Precision])
        
        avg_AdjustedRandScore = sum(AdjustedRandScore_t)/len(AdjustedRandScore_t)
        avg_Precision = sum(Precision_t)/len(Precision_t)
    
    return avg_AdjustedRandScore, avg_Precision



def NPinFileSelection(NPinFile_case_Type, reservedTermsTotal):
    case_DT_1stLDA=[]
    for file in NPinFile_case_Type:
        case_DT_1stLDA_t=[]
        for item in file:
            if item in reservedTermsTotal:
                case_DT_1stLDA_t.append(item)
        case_DT_1stLDA.append(case_DT_1stLDA_t)  
        
    return case_DT_1stLDA
    


def num_GS_terms(newcorpus):
    temp1 = np.unique(l3_plain(hypernym_1_N_hyponym_human3))
    temp2 = np.unique(l2_plain(newcorpus))
    counter = 0
    for item in temp2:
        if item in temp1:
            counter += 1   
    return counter
    
