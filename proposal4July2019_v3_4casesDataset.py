#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:03:50 2019

duplicate from <proposal4March2019_v1.py>

difference:
    the path for input is different (update to the new variables) 
    hypernym_1_N_hyponym_human3 --> hypernym_1_N_hyponym_human4 --> hypernym_1_N_hyponym_human4_train
    

@author: ziwei
"""
import numpy as np
import os

import spacy 
from spacy.lang.en.stop_words import STOP_WORDS

import gensim


# case 1:
core_lables3_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human4'+'.npy')
hypernym_1_N_hyponym_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4'+'.npy')
#l3_frequency(hypernym_1_N_hyponym_human4) #2776
hypernym_1_N_hyponym_human4_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy')
#l3_frequency(hypernym_1_N_hyponym_human4_train) #2222

name_subdomain11 = ['Algorithm design','Bioinformatics','Computer graphics',
                  'Computer programming','Cryptography','Data structures',
                  'Distributed computing','Machine learning','Operating systems',
                  'Software engineering','network security'] #11
#!!! new version 
name_subdomain11_v2 = ['Computer graphics','Machine learning','network security',
                    'Cryptography','Operating systems', 'Software engineering',
                    'Distributed computing','Algorithm design','Computer programming',
                    'Data structures','Bioinformatics'] #11



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

# for evaluation: keep these GS terms from elimination from filtering
GS_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy') 
GS_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy') 
keepToken = l3_plain(GS_train)
keepToken.extend(l3_plain(GS_test))
#len(keepToken) #2776



###!!!!---------------------------case1: only NPs with syntactic roles in files------------------------------
# input: 
NPs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPs_SubjObj_PerfileList) | 113,140 extracted NPs
VBs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/VBs_SubjObj_PerfileList'+'.npy')
# l2_frequency(VBs_SubjObj_PerfileList) | 113,140 extracted NPs


################## goldNPs_set
### problem: Nps_rep_PerfileList_case1 was always overwitten by Nps_rep_PerfileList_case2. 

#temp =l3_plain(hypernym_1_N_hyponym_human4)
#Nps_rep_PerfileList_case1 = []
#Nps_rep_PerfileList_case2 = [] # based on case1, add verbs inside (excluding stopwords)
#for ind_file, file in enumerate(NPs_SubjObj_PerfileList):
#    temp2 = [] # store nps
#    temp3 = [] # store verbs
#    for ind_item, item in enumerate(file):
#        if item in temp:
#            temp2.append(item)
#            temp3.append(VBs_SubjObj_PerfileList[ind_file][ind_item])
#    # delete stopwords in verbs list
#    temp3 = [i for i in temp3 if i not in STOP_WORDS]
#    Nps_rep_PerfileList_case1.append(temp2) 
#    
#    Nps_rep_PerfileList_case2.append(temp2) 
#    Nps_rep_PerfileList_case2[ind_file].extend(temp3) 
    

temp =l3_plain(hypernym_1_N_hyponym_human4)
Nps_rep_PerfileList_case1 = []
Nps_rep_PerfileList_case2 = [] # based on case1, add verbs inside (excluding stopwords)
for ind_file, file in enumerate(NPs_SubjObj_PerfileList):
    temp2 = [] # store nps
    for ind_item, item in enumerate(file):
        if item in temp:
            temp2.append(item)
    Nps_rep_PerfileList_case1.append(temp2) 
for ind_file, file in enumerate(NPs_SubjObj_PerfileList):
    temp2 = [] # store nps
    temp3 = [] # store verbs
    for ind_item, item in enumerate(file):
        if item in temp:
            temp2.append(item)
            temp3.append(VBs_SubjObj_PerfileList[ind_file][ind_item])
    # delete stopwords in verbs list
    temp3 = [i for i in temp3 if i not in STOP_WORDS]   
    Nps_rep_PerfileList_case2.append(temp2) 
    Nps_rep_PerfileList_case2[ind_file].extend(temp3)    
    

#l2_frequency(Nps_rep_PerfileList_case1) #2383
Nps_rep_PerfileList_case1_t = [i for i in Nps_rep_PerfileList_case1 if i != []] #1726. # of None empty lists
#len(np.unique(l2_plain(Nps_rep_PerfileList_case1))) #1828 (less than 2776,the correct number as we set up the <hypernym_1_N_hyponym_human4>)

#l2_frequency(Nps_rep_PerfileList_case2) #4503
Nps_rep_PerfileList_case2_t = [i for i in Nps_rep_PerfileList_case2 if i != []] #1726. # of None empty lists
#len(np.unique(l2_plain(Nps_rep_PerfileList_case2))) #2177 


np.save(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case1', Nps_rep_PerfileList_case1)
#Nps_rep_PerfileList_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case1'+'.npy')
# l2_frequency(Nps_rep_PerfileList_case1) | 2383 extracted keywords, 1726 files

np.save(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case1_t', Nps_rep_PerfileList_case1_t)
#Nps_rep_PerfileList_case1_t = np.load(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case1_t'+'.npy')
# l2_frequency(Nps_rep_PerfileList_case1_t) | 2383 extracted keywords, 1726 files

np.save(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case2', Nps_rep_PerfileList_case2)
#Nps_rep_PerfileList_case2 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case2'+'.npy')
# l2_frequency(Nps_rep_PerfileList_case2) | 4503 extracted keywords, 1726 files

np.save(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case2_t', Nps_rep_PerfileList_case2_t)
#Nps_rep_PerfileList_case2_t = np.load(os.getcwd()+'/intermediateRes/FourCases_res/Nps_rep_PerfileList_case2_t'+'.npy')
# l2_frequency(Nps_rep_PerfileList_case2_t) | 4503 extracted keywords, 1726 files

################## goldNPs_set_rep | replacedByCentralTerms2()

################## goldNPs_set_keywords | goldNPs_set_keywords_rep
# only focus on files >= 2
# add keywords for each file



### 1). extract defined doamins and keywords   #!!! new version 
#purpose: to ge the uniformed format keywords lists

# input: | wosData_cs["Y2"].values |

# output: | keywords_list | delete head space and tail space
#         | keywords_list2 | pre-process into uniformed format
#         | keywords_list3 | only keep keywords that are appeared in our defined table
# keywords_list3: stored the keywords for each file

domain_list = [] #6514
for y2 in wosData_cs["Y2"].values:
    if y2 in [0,5,6]:
        domain_list.extend([0])
    elif y2 in [7,10,12]:
        domain_list.extend([7])
    elif y2 in [8,11,15]:
        domain_list.extend([8])
    else:
        domain_list.extend([y2]) 

keywords_list=[]
for file in wosData_cs['keywords'].values:
    keywords_list_t=[]
    keywords_split = re.split(r';',file)
#    keywords_list1.append(temp)
    for item in keywords_split:
        temp2 = re.sub(r'^ +',r'',item) #delete head space
        temp3 = re.sub(r' +$',r'',temp2) #delete tail space            
        keywords_list_t.append(temp3) 
    keywords_list.append(keywords_list_t)

#to pre-process "keywords" into our defined format
keywords_list2 = definedFormat2layer(keywords_list)

#only keep keywords that are appeared in our defined table
keywords_list3 = copy.deepcopy(keywords_list2)
keypwords_gold = l2_plain(core_lables3_human4)
for inx_file, file in enumerate(keywords_list2):
    for ele in file:
        if ele not in keypwords_gold:
            keywords_list3[inx_file].remove(ele)


np.save(os.getcwd()+'/intermediateRes/FourCases_res/keywords_list3', keywords_list3)
#keywords_list3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/keywords_list3'+'.npy')
# l2_frequency(keywords_list3) | 5727 extracted keywords, 6514 files




### 2). find subset by defining the restriction of keywords' addition
# RULES: only when files >=2 && only when # file > # keywords

size_keywords = l2_frequency_files(keywords_list3)
size_list = l2_frequency_files(Nps_rep_PerfileList_case1)

# the index of subsets
inx_subset_case1 = []
for inx_k, item in enumerate(size_keywords):
    if item > 1 and item < size_list[inx_k] and item!= 0:
        inx_subset_case1.append(inx_k)        
 

np.save(os.getcwd()+'/intermediateRes/FourCases_res/inx_subset_case1', inx_subset_case1)
#inx_subset_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/inx_subset_case1'+'.npy')
# len(inx_subset_case1)  #37 files



### 3). 
#       input: | inx_subset_case1 | index of extracted GS syntactic NP files according to rules
#              | Nps_rep_PerfileList_case1 | syntactic NP files
#              | core_lables3_human4 | keywords or central terms for each domains

#       output:
#              | goldNPs_set_case1 | the extracted GS syntactic NP files according to rules
#              --> "core concept replacement"
#              | goldNPs_set_rep_case1 | the keywords or central terms in files are replaced by doamin names (core concepts)
#              --> "sub-domain knowledge replacement" | method1 | & | method2 |
#              | goldNPs_set_keywords_case1 | add keyowrds or central terms in the tail of each file
#              | goldNPs_set_keywords_rep_case1 |  replace centralTerms by domain name (core concepts)


goldNPs_set_case1 = [Nps_rep_PerfileList_case1[i] for i in inx_subset_case1] #total NPs is 130, 37 files
goldNPs_set_rep_case1 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case1, core_lables3_human4)[0]
# 6 occurrence, 5 unique terms
    

# all replaced terms in gold sets 
rep_goldset1 = [] #18
for inx_file, file in enumerate(goldNPs_set_case1):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case1[inx_file][inx_item]:
            rep_goldset1.append(goldNPs_set_rep_case1[inx_file][inx_item])
len(np.unique(rep_goldset1)) #15


# add the keywords at the tail of goldNP files
goldNPs_set_keywords_case1 = []
for i in inx_subset_case1:
    temp = copy.deepcopy(Nps_rep_PerfileList_case1[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case1.append(temp)   
#l2_frequency(goldNPs_set_keywords_case1)-l2_frequency(goldNPs_set_case1) 
# 209 | 130 | the diff: 79 (the addition of keywords)


# 53 times to replace centralTerms by domain name (core concepts)
goldNPs_set_keywords_rep_case1 =[] #37
counter = 0 #79
for inx_i, i in enumerate(inx_subset_case1):
    
    temp = copy.deepcopy(goldNPs_set_rep_case1[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    temp.extend([string_t] * len(keywords_list3[i]))
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case1.append(temp)

'''
Nps_rep_PerfileList_case1[42]
['phase based optical flow algorithm','unscented Kalman filter','unscented Kalman filter']

goldNPs_set_keywords_case1[0]
['phase based optical flow algorithm','unscented Kalman filter','unscented Kalman filter','motion magnification','computer vision']

goldNPs_set_keywords_rep_case1[0]
['phase based optical flow algorithm','unscented Kalman filter','unscented Kalman filter','Computer graphics','Computer graphics']
'''  

np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case1', goldNPs_set_case1)
#goldNPs_set_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case1'+'.npy')
# l2_frequency(goldNPs_set_case1) | 130 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case1', goldNPs_set_rep_case1)
#goldNPs_set_rep_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case1'+'.npy')
# l2_frequency(goldNPs_set_rep_case1) | 130 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case1', goldNPs_set_keywords_case1)
#goldNPs_set_keywords_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case1'+'.npy')
# l2_frequency(goldNPs_set_keywords_case1) | 209 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case1', goldNPs_set_keywords_rep_case1)
#goldNPs_set_keywords_rep_case1 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case1'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case1) | 209 nps


############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case1        
termsPerfileList =  goldNPs_set_rep_case1
termsPerfileList =  goldNPs_set_keywords_case1        
termsPerfileList =  goldNPs_set_keywords_rep_case1

filter_no_below = 0
filter_no_above = 1
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)


#NPswithLabel, LabelCluster, LabelClass, AdjustedRandScore, Precision = Eval_label_assign(hypernym_1_N_hyponym_human4, TermTps)

GS_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy') 
GS_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy') 
res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)


#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/TermTps_c1t4',TermTps)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/NPswithLabel_c1t4',NPswithLabel)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelCluster_c1t4',LabelCluster)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelClass_c1t4',LabelClass)


    



# case 2:

###### rebuild the corpus only contains gold standard terms
'''
        input: 
            | Nps_rep_PerfileList_case2 | add verbs on the basis of Nps_rep_PerfileList_case1
#            | NPs_SubjObj_PerfileList | syntactic NPs 
#            | name_subdomain11 | core concpets
#            | core_lables3_human4 | centralTerms; keywords lists
#            | nounlistinFile2 | source files for all NPs
#            | Nps_rep_PerfileList_case1 | borrow from case 1: the GS syntactoc NPs in files according to rules, the sequence of whole file

'''
# try1: we have function <replacedByCentralTerms4> to do everything and also verbs, but the quatities are not equal. thus omit
    

################## goldNPs_set_keywords_case2 | goldNPs_set_keywords_rep_case2

### 1). to reuse the | domain_list | keywords_list3 | 
            
### 2). find subset by defining the restriction of keywords' addition
### case1 and case2 should use the same subsets 
inx_subset_case2 = inx_subset_case1


### 3). # | goldNPs_set_case2 | goldNPs_set_rep_case2 | goldNPs_set_keywords_case2 | goldNPs_set_keywords_rep_case2 | 
# before training LDA, try to provide all files
goldNPs_set_case2 = [Nps_rep_PerfileList_case2[i] for i in inx_subset_case2]
goldNPs_set_rep_case2 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case2, core_lables3_human4)[0]
# 6 occurrence, 5 unique terms | the same as case1 (correct)

# all replaced terms in gold sets (different from case1, because verbs are considered as verbs in GS)
rep_goldset = [] #36
for inx_file, file in enumerate(goldNPs_set_case2):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case2[inx_file][inx_item]:
            rep_goldset.append(goldNPs_set_rep_case2[inx_file][inx_item])
#len(np.unique(rep_goldset)) #31



goldNPs_set_keywords_case2 = []
for i in inx_subset_case2:
    temp = copy.deepcopy(Nps_rep_PerfileList_case2[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case2.append(temp)
#l2_frequency(goldNPs_set_keywords_case2)-l2_frequency(goldNPs_set_case2) 
# 326 | 247 | the diff: 79 (the addition of keywords)

goldNPs_set_keywords_rep_case2 =[] #37
counter = 0 # 79
for inx_i, i in enumerate(inx_subset_case2):
    
    temp = copy.deepcopy(goldNPs_set_rep_case2[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    print(string_t)
    temp.extend([string_t] * len(keywords_list3[i]))
    print(temp)
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case2.append(temp)
    

np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case2', goldNPs_set_case2)
#goldNPs_set_case2 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case2'+'.npy')
# l2_frequency(goldNPs_set_case2) | 247 nps+ verbs
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case2', goldNPs_set_rep_case2)
#goldNPs_set_rep_case2 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case2'+'.npy')
# l2_frequency(goldNPs_set_rep_case2) | 247 nps+verbs
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case2', goldNPs_set_keywords_case2)
#goldNPs_set_keywords_case2 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case2'+'.npy')
# l2_frequency(goldNPs_set_keywords_case2) | 326 nps+verbs
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case2', goldNPs_set_keywords_rep_case2)
#goldNPs_set_keywords_rep_case2 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case2'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case2) | 326 nps+verbs



############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case2        
termsPerfileList =  goldNPs_set_rep_case2
termsPerfileList =  goldNPs_set_keywords_case2        
termsPerfileList =  goldNPs_set_keywords_rep_case2

filter_no_below = 0
filter_no_above = 1
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

GS_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy') 
GS_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy') 
res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)


#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/TermTps_c2t2',TermTps)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/NPswithLabel_c2t2',NPswithLabel)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelCluster_c2t2',LabelCluster)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelClass_c2t2',LabelClass)






# case 3:
NPs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPs_SubjObj_PerfileList) | 113140 extracted NPs
Nps_rep_PerfileList_case3 =  NPs_SubjObj_PerfileList                                  


################## goldNPs_set_keywords_case3 | goldNPs_set_keywords_rep_case3
# only focus on files >= 10
# add keywords for each file
### 1). to reuse the | domain_list | keywords_list3 | 
            
### 2). find subset by defining the restriction of keywords' addition
# only when files >=10 && only when # file > # keywords
size_keywords = l2_frequency_files(keywords_list3)
size_list = l2_frequency_files(Nps_rep_PerfileList_case3)

# the index of subsets
inx_subset_case3 = []
for inx_k, item in enumerate(size_list):
    if item > 9 and item > size_keywords[inx_k]:
        inx_subset_case3.append(inx_k)        
len(inx_subset_case3)  #5744  


### 3). # | goldNPs_set_case3 | goldNPs_set_rep_case3 | goldNPs_set_keywords_case3 | goldNPs_set_keywords_rep_case3 | 
# before training LDA, try to provide all files
goldNPs_set_case3 = [Nps_rep_PerfileList_case3[i] for i in inx_subset_case3] #total NPs is 107737
goldNPs_set_rep_case3 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case3, core_lables3_human4)[0] ##total NPs is 107737

# all replaced terms in gold sets 
rep_goldset3 = [] #20008 
for inx_file, file in enumerate(goldNPs_set_case3):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case3[inx_file][inx_item]:
            rep_goldset3.append(goldNPs_set_rep_case3[inx_file][inx_item])
len(np.unique(rep_goldset3)) #10259 



goldNPs_set_keywords_case3 = []
for i in inx_subset_case3:
    temp = copy.deepcopy(Nps_rep_PerfileList_case3[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case3.append(temp)
#l2_frequency(goldNPs_set_keywords_case3)-l2_frequency(goldNPs_set_case3) 
# 112832 | 107737 | the diff: 5095 (the addition of keywords)

# 5095 words occurrence, 
goldNPs_set_keywords_rep_case3 =[]
counter = 0
for inx_i, i in enumerate(inx_subset_case3):
    
    temp = copy.deepcopy(goldNPs_set_rep_case3[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    print(string_t)
    temp.extend([string_t] * len(keywords_list3[i]))
    print(temp)
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case3.append(temp)



np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case3', goldNPs_set_case3)
#goldNPs_set_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case3'+'.npy')
# l2_frequency(goldNPs_set_case3) | 107737 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case3', goldNPs_set_rep_case3)
#goldNPs_set_rep_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case3'+'.npy')
# l2_frequency(goldNPs_set_rep_case3) | 107737 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case3', goldNPs_set_keywords_case3)
#goldNPs_set_keywords_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case3'+'.npy')
# l2_frequency(goldNPs_set_keywords_case3) | 112832 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case3', goldNPs_set_keywords_rep_case3)
#goldNPs_set_keywords_rep_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case3'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case3) | 112832 nps




############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case3        
termsPerfileList =  goldNPs_set_rep_case3
termsPerfileList =  goldNPs_set_keywords_case3       
termsPerfileList =  goldNPs_set_keywords_rep_case3

filter_no_below = 5
filter_no_above = 0.5
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
# without pruning, 50902
# len(NPsdictionary) 
'''
filter_no_below 3, other 0.5 | 4909
filter_no_below 5, other 0.5 | 3413
'''
ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)








# case4 # all NPs and all verbs 

NPsVBs_SubjObj_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_SubjObj_PerfileList'+'.npy')
# l2_frequency(NPsVBs_SubjObj_PerfileList) | 205414 extracted NPs 
Nps_rep_PerfileList_case4 = NPsVBs_SubjObj_PerfileList


################## goldNPs_set_keywords_case4 | goldNPs_set_keywords_rep_case4
# only focus on files >= 10
# add keywords for each file
### 1). to reuse the | domain_list | keywords_list3 | 
            
### 2). find subset by defining the restriction of keywords' addition
# only when files >=10 && only when # file > # keywords
# the index of subsets
# use the subset from | inx_subset_case3 |
inx_subset_case4 = inx_subset_case3

### 3). # | goldNPs_set_case4 | goldNPs_set_rep_case4 | goldNPs_set_keywords_case4 | goldNPs_set_keywords_rep_case4 | 
# before training LDA, try to provide all files
goldNPs_set_case4 = [Nps_rep_PerfileList_case4[i] for i in inx_subset_case4] #total NPs is 195687
goldNPs_set_rep_case4 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case4, core_lables3_human4)[0]  #total NPs is 195687


# all replaced terms in gold sets 
rep_goldset4 = [] #41584
for inx_file, file in enumerate(goldNPs_set_case4):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case4[inx_file][inx_item]:
            rep_goldset4.append(goldNPs_set_rep_case4[inx_file][inx_item])
len(np.unique(rep_goldset4)) #11308


goldNPs_set_keywords_case4 = []
for i in inx_subset_case4:
    temp = copy.deepcopy(Nps_rep_PerfileList_case4[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case4.append(temp)
#l2_frequency(goldNPs_set_keywords_case4)-l2_frequency(goldNPs_set_case4) 
# 200782 | 195687 | the diff: 5095 (the addition of keywords)

# 5095 words occurrence
goldNPs_set_keywords_rep_case4 =[]
counter = 0
for inx_i, i in enumerate(inx_subset_case4):
    
    temp = copy.deepcopy(goldNPs_set_rep_case4[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    temp.extend([string_t] * len(keywords_list3[i]))
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case4.append(temp)
 
    
    
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case4', goldNPs_set_case4)
#goldNPs_set_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case4'+'.npy')
# l2_frequency(goldNPs_set_case4) | 195687 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case4', goldNPs_set_rep_case4)
#goldNPs_set_rep_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case4'+'.npy')
# l2_frequency(goldNPs_set_rep_case4) | 195687 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case4', goldNPs_set_keywords_case4)
#goldNPs_set_keywords_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case4'+'.npy')
# l2_frequency(goldNPs_set_keywords_case4) | 200782 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case4', goldNPs_set_keywords_rep_case4)
#goldNPs_set_keywords_rep_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case4'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case4) | 200782 nps
   

############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case4        
termsPerfileList =  goldNPs_set_rep_case4
termsPerfileList =  goldNPs_set_keywords_case4        
termsPerfileList =  goldNPs_set_keywords_rep_case4


filter_no_below = 5
filter_no_above = 0.5
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)

#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/TermTps_c4t3',TermTps)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/NPswithLabel_c4t3',NPswithLabel)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelCluster_c4t3',LabelCluster)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelClass_c4t3',LabelClass)



# case 5:
NpsPerfileList = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')
# l2_frequency(NpsPerfileList) | 291,488 extracted NPs
Nps_rep_PerfileList_case5 =  NpsPerfileList                                  


################## goldNPs_set_keywords_case5 | goldNPs_set_keywords_rep_case5
# only focus on files >= 10
# add keywords for each file
### 1). to reuse the | domain_list | keywords_list3 | 
            
### 2). find subset by defining the restriction of keywords' addition
# only when files >=10 && only when # file > # keywords
size_keywords = l2_frequency_files(keywords_list3)
size_list = l2_frequency_files(Nps_rep_PerfileList_case5)

# the index of subsets
inx_subset_case5 = []
for inx_k, item in enumerate(size_list):
    if item > 9 and item > size_keywords[inx_k]:
        inx_subset_case5.append(inx_k)        
len(inx_subset_case5)  #6492  


### 3). # | goldNPs_set_case5 | goldNPs_set_rep_case5 | goldNPs_set_keywords_case5 | goldNPs_set_keywords_rep_case5 | 
# before training LDA, try to provide all files
goldNPs_set_case5 = [Nps_rep_PerfileList_case5[i] for i in inx_subset_case5] #total NPs is 291317
goldNPs_set_rep_case5 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case5, core_lables3_human4)[0] ##total NPs is 291317

# all replaced terms in gold sets 
rep_goldset3 = [] #1637751 
for inx_file, file in enumerate(goldNPs_set_case5):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case5[inx_file][inx_item]:
            rep_goldset3.append(goldNPs_set_rep_case5[inx_file][inx_item])
len(np.unique(rep_goldset3)) #51566 



goldNPs_set_keywords_case5 = []
for i in inx_subset_case5:
    temp = copy.deepcopy(Nps_rep_PerfileList_case5[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case5.append(temp)
#l2_frequency(goldNPs_set_keywords_case5)-l2_frequency(goldNPs_set_case5) 
# 297029 | 291317 | the diff: 5712 (the addition of keywords)

# 5712 words occurrence, 
goldNPs_set_keywords_rep_case5 =[]
counter = 0
for inx_i, i in enumerate(inx_subset_case5):
    
    temp = copy.deepcopy(goldNPs_set_rep_case5[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    print(string_t)
    temp.extend([string_t] * len(keywords_list3[i]))
    print(temp)
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case5.append(temp)



np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case5', goldNPs_set_case5)
#goldNPs_set_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case5'+'.npy')
# l2_frequency(goldNPs_set_case5) | 291317 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case5', goldNPs_set_rep_case5)
#goldNPs_set_rep_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case5'+'.npy')
# l2_frequency(goldNPs_set_rep_case5) | 291317 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case5', goldNPs_set_keywords_case5)
#goldNPs_set_keywords_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case5'+'.npy')
# l2_frequency(goldNPs_set_keywords_case5) | 297029 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case5', goldNPs_set_keywords_rep_case5)
#goldNPs_set_keywords_rep_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case5'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case5) | 297029 nps




############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case5        
termsPerfileList =  goldNPs_set_rep_case5
termsPerfileList =  goldNPs_set_keywords_case5       
termsPerfileList =  goldNPs_set_keywords_rep_case5

filter_no_below = 5
filter_no_above = 0.5
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
ixTermTps, TermTps, highest_relev2, topic_term_prob, topic_term_relev = clusterDistribution4(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
res_predict, res_nps_test, test_res = Eval_label_assign2(GS_train, GS_test, TermTps, ixTermTps, topic_term_prob, NPsdictionary)

#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/TermTps_c3t3',TermTps)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/NPswithLabel_c3t3',NPswithLabel)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelCluster_c3t3',LabelCluster)
#np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelClass_c3t3',LabelClass)



# case6 # all NPs and all verbs 

NPsVBs_PerfileList = np.load(os.getcwd()+'/intermediateRes/NPsVBs_PerfileList'+'.npy')
# l2_frequency(NPsVBs_PerfileList) | 424,637 extracted NPs 
Nps_rep_PerfileList_case6 = NPsVBs_PerfileList


################## goldNPs_set_keywords_case6 | goldNPs_set_keywords_rep_case6
# only focus on files >= 10
# add keywords for each file
### 1). to reuse the | domain_list | keywords_list3 | 
            
### 2). find subset by defining the restriction of keywords' addition
# only when files >=10 && only when # file > # keywords
# the index of subsets
# use the subset from | inx_subset_case5 |
inx_subset_case6 = inx_subset_case5

### 3). # | goldNPs_set_case6 | goldNPs_set_rep_case6 | goldNPs_set_keywords_case6 | goldNPs_set_keywords_rep_case6 | 
# before training LDA, try to provide all files
goldNPs_set_case6 = [Nps_rep_PerfileList_case6[i] for i in inx_subset_case6] #total NPs is 424362
goldNPs_set_rep_case6 = replacedByCentralTerms2(name_subdomain11, goldNPs_set_case6, core_lables3_human4)[0]  #total NPs is 424362


# all replaced terms in gold sets 
rep_goldset4 = [] #2103532
for inx_file, file in enumerate(goldNPs_set_case6):
    for inx_item, item in enumerate(file):
        if item != goldNPs_set_rep_case6[inx_file][inx_item]:
            rep_goldset4.append(goldNPs_set_rep_case6[inx_file][inx_item])
len(np.unique(rep_goldset4)) #53313


goldNPs_set_keywords_case6 = []
for i in inx_subset_case6:
    temp = copy.deepcopy(Nps_rep_PerfileList_case6[i])
    temp.extend(keywords_list3[i])
    goldNPs_set_keywords_case6.append(temp)
#l2_frequency(goldNPs_set_keywords_case6)-l2_frequency(goldNPs_set_case6) 
# 430074 | 424362 | the diff: 5712 (the addition of keywords)

# 5712 words occurrence
goldNPs_set_keywords_rep_case6 =[]
counter = 0
for inx_i, i in enumerate(inx_subset_case6):
    
    temp = copy.deepcopy(goldNPs_set_rep_case6[inx_i])
    string_t = name_subdomain11_v2[list(np.unique(domain_list)).index(domain_list[i])]
    temp.extend([string_t] * len(keywords_list3[i]))
    counter = counter + len(keywords_list3[i])
    goldNPs_set_keywords_rep_case6.append(temp)
 
    
    
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case6', goldNPs_set_case6)
#goldNPs_set_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case6'+'.npy')
# l2_frequency(goldNPs_set_case6) | 424362 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case6', goldNPs_set_rep_case6)
#goldNPs_set_rep_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case6'+'.npy')
# l2_frequency(goldNPs_set_rep_case6) | 424362 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case6', goldNPs_set_keywords_case6)
#goldNPs_set_keywords_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case6'+'.npy')
# l2_frequency(goldNPs_set_keywords_case6) | 430074 nps
np.save(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case6', goldNPs_set_keywords_rep_case6)
#goldNPs_set_keywords_rep_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case6'+'.npy')
# l2_frequency(goldNPs_set_keywords_rep_case6) | 430074 nps
   




############## train LDA and evaluate (one times) ##########
termsPerfileList =  goldNPs_set_case6        
termsPerfileList =  goldNPs_set_rep_case6
termsPerfileList =  goldNPs_set_keywords_case6        
termsPerfileList =  goldNPs_set_keywords_rep_case6

filter_no_below = 0
filter_no_above = 1
tp = 50

lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp, KEEP_TOKEN = keepToken) 
ixTermTps, TermTps, highest_relev2 = clusterDistribution2(bow_corpus, NPsdictionary, lda_model)

NPswithLabel, LabelCluster, LabelClass, AdjustedRandScore, Precision = Eval_label_assign(hypernym_1_N_hyponym_human4, TermTps)

np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/TermTps_c4t3',TermTps)
np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/NPswithLabel_c4t3',NPswithLabel)
np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelCluster_c4t3',LabelCluster)
np.save('/Users/zoe/anaconda3/LDA experiments/paper_example/LabelClass_c4t3',LabelClass)
# output excel in example file

# goldNPs_set :  0.01776253531924366 | 0.35156676090064665
# goldNPs_set_rep: 0.006252561697067859 | 0.2911635436257259
# goldNPs_set_keywords: 0.01040577940608193 | 0.3307329207889415
# goldNPs_set_keywords_rep: 0.008364036474065863 | 0.30393492040627773 

############## train LDA and evaluate (10 times) | TIME CONSUMING ##########
termsPerfileList =  goldNPs_set_case6        
termsPerfileList =  goldNPs_set_rep_case6
termsPerfileList =  goldNPs_set_keywords_case6       
termsPerfileList =  goldNPs_set_keywords_rep_case6

filter_no_below = 0
filter_no_above = 1
tp = 50

avg_AdjustedRandScore, avg_Precision = train_eval_10(termsPerfileList, filter_no_below, filter_no_above, tp)
# update 23th March 2019
# goldNPs_set :  0.013158462241870028| 0.31257450600461895
# goldNPs_set_rep: 0.013787152411938666 | 0.32177515990576366
# goldNPs_set_keywords: 0.016085354913747414 | 0.3378564994000258
# goldNPs_set_keywords_rep: 0.014214225483310649 | 0.3214487237462085 