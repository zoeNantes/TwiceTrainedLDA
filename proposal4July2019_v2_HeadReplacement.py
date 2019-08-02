#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:39:20 2019
# target: head replacement upon data

# reedit from: <proposal4Jan2019_v7_headKeywordList> 
# differece between last one:
        change path from original path into new dictionary path 

# input:
        
        NpsPerfileList = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')
        nounlistinFile2 = np.load(os.getcwd()+'/intermediateRes/nounlistinFile2'+'.npy')


# output: 
        # 17 lists, 11 lists have value. Each list includes around top 100 terms, excluding general terms and cross-domain terms
        core_lables3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3'+'.npy')
        
        # 11 lists. Each list includes around top 50 terms evaluated by human from core_lables3 
        core_lables3_human3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human3'+'.npy')
        # delete duplicate terms inside each domain
        core_lables3_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human4'+'.npy')
        
        # 11 lists. roughly pattern matching to find hyponyms of keywords: | 3 layers | domain -> keyword-> hyponyms of keywords |   
        lables_centralTerms_hyponyms = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/lables_centralTerms_hyponyms'+'.npy')
        # 11 lists. the hyponums of keywords after human evaluation
        hypernym_1_N_hyponym_human3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human3'+'.npy')

        hypernym_1_N_hyponym_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4'+'.npy')
        # the way to find our desired GS 
            #1). central terms (keywords) as hypernyms (top 100, delete duplicated, delete unrelated for human)
            #2). to find long strings from text which are roughly matched to central terms -->(hypernym_1_N_hyponym_human3)
            #3). to find roughly matched in-text extracted NPs with hyponyms from hypernym_1_N_hyponym_human3
            #4). to enlarge the GS with rough matched intext NPs

        GS are divided into two parts (trainset: 80%; testset: 20%)
        hypernym_1_N_hyponym_human4_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy')
        hypernym_1_N_hyponym_human4_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy')
       


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

import xlsxwriter

import os




################################## Head replacement in Target Data ################################


# 1. load data 
file_wos_location = os.getcwd()+'/SourceData/Data.xlsx'
wosData = pd.read_excel(file_wos_location)
wosData_cs = wosData[wosData.Y1==0]   #[6514 rows x 7 columns]

  
# --------------------------------keywords processing--------------------------    
# delete words that only appear once 
# return two list, one with unique words, one with full words


#################### to extract all keywords for each domain
# output: | subdomain_keywords_list2 | subdomain_keywords_list3 |

name_subdomain = [' Algorithm design  ',' Bioinformatics  ',' Computer graphics  ',
                  ' Computer programming  ',' Computer vision  ',' Cryptography  ',
                  ' Data structures  ',' Distributed computing  ',' Image processing  ',
                  ' Machine learning  ',' Operating systems  ',' Parallel computing  ',
                  ' Relational databases  ',' Software engineering  ',' Structured Storage  ',
                  ' Symbolic computation  ',' network security  '] #17

name_subdomain11 = ['Algorithm design','Bioinformatics','Computer graphics',
                  'Computer programming','Cryptography','Data structures',
                  'Distributed computing','Machine learning','Operating systems',
                  'Software engineering','network security'] #11
#!!! new version 
name_subdomain11_v2 = ['Computer graphics','Machine learning','network security',
                    'Cryptography','Operating systems', 'Software engineering',
                    'Distributed computing','Algorithm design','Computer programming',
                    'Data structures','Bioinformatics'] #11


def del_by_frequency(temp): #given original array, return deleted array and amount of deleted elements
    _, idx, count = np.unique(temp, return_inverse=True,return_counts=True)
    temp_array = np.array(temp)
    return temp_array[np.in1d(idx,np.where(count>=2)[0])], len(np.where(count<2)[0])

subdomain_keywords_list2=[] # unique 
subdomain_keywords_list3=[] # with duplicate terms

for item in name_subdomain:
    
    #find keywords that have same subdomain
    df_keywords = wosData_cs.loc[lambda wosData_cs: wosData_cs["area"] == item].keywords  
    df_keywords_val = df_keywords.values

    #split keywords into single item and delete unmeaningful space
    df_keywords_list = []
    df_keywords_list_witho_space = []
    df_keywords_list_witho_space_uniq = []
    for i in range(len(df_keywords)):
        temp = re.split(r';',df_keywords_val[i])
        #    temp_flat = [y for x in temp for y in x] 
        df_keywords_list.extend(temp)
        
        # delete head and tail space
        for keywds in df_keywords_list:
            temp2 = re.sub(r'^ +',r'',keywds) #delete head space
            temp3 = re.sub(r' +$',r'',temp2) #delete tail space            
            df_keywords_list_witho_space.append(temp3)
    
    # delete elements whose frequency is once
    df_keywords_list_witho_space, c = del_by_frequency(df_keywords_list_witho_space)
    print(c)
                
    subdomain_keywords_list2.append(np.unique(df_keywords_list_witho_space))  
    subdomain_keywords_list3.append(df_keywords_list_witho_space) 

#################### to pre-process "keywords" into our defined format
# output: | nounPrase_2 |
    
noisy_pos_tags = ['SYM','NUM','PUNCT','SPACE','SCONJ','X']

nounPrase_2=[]
for num_lable in range(len(subdomain_keywords_list2)):
    nounPrase_1=[]
    print(num_lable)
    for num_ele in range(len(subdomain_keywords_list2[num_lable])):
      
        temp = str(subdomain_keywords_list2[num_lable][num_ele])
        doc = nlp(temp)

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

##################### add the frequency to the new formatted NPs in nounPrase_2
# input:
# | nounPrase_2: new NPs
# | subdomain_keywords_list3: original frequency
# output:
# | dict0, dict1, ...,dict16 (keywords with frequency)
for nb_label in range(len(subdomain_keywords_list3)):
    
    #3.1 make dictionary
    nb_counts = list(np.unique(subdomain_keywords_list3[nb_label], return_counts=True)[1])
    c = dict(zip(nounPrase_2[nb_label], nb_counts))  # error, the sam value will be overwrited to be the last one
     
    #3.2 find terms with multiple occurrence, and add their previous value to dict
    temp1= list(Counter(nounPrase_2[nb_label]).values())  #all values in list 
    temp2 = [] #temp2 is duplicated words
    
    for index, i in enumerate(temp1):
        if i > 1:
            temp2.append(list(Counter(nounPrase_2[nb_label]))[index]) 
            
    #3.3 find index of previous duplicated words and plus the value
    for dup_item in temp2:  
        temp3 = []
        for index, item in enumerate(nounPrase_2[nb_label]):
            if item == dup_item:
                temp3.append(index) #temp3 is lsit of index for duplicated words
        temp3.pop() #delete the last elements in list
        
        # add the frequency to dict
        for i in temp3: 
            c[dup_item] += nb_counts[i]
    
    globals()['dict%s' % nb_label] = c   #give dynymaic names to variables

###################### verify: "wireless sensor network" --> correct !!!
for index, i in enumerate(nounPrase_2[16]):
    if i == "wireless sensor network":
        print(index)

nb_counts[897] #294
nb_counts[904] #535
nb_counts[905] #323
nb_counts[1681] #779


#!!! -------------------------manually keywords selection--------------------------------- 

# idea: most common 100 terms to extract keyterms used as central terms of head replacement
# head replacment list generation with human intervention
# find central words in most 100 frequency terms


# "network security"
sorted_by_value = sorted(dict16.items(), key=lambda kv: kv[1], reverse= True)
keywords100_16 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_16 = ['security', 'simulation', 'entropy', 'time series', 'complex network', 
 'exploit', 'weakness', 'vulnerability', 'robustness', 'big datum',
 'energy efficiency', 'capacity', 'measure', 'risk analysis']
# delete cross-domain terms
crosDom_16 = ['cloud compute', 'cloud computing', 'machine learn', 'cloud model', 'neural network',
 'k mean', 'cluster', 'genetic algorithm','finite automata', 'graph theory', 'mapreduce']


# "symbolic computation"
sorted_by_value = sorted(dict15.items(), key=lambda kv: kv[1], reverse= True)
keywords100_15 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_15 = ['exact solution','', 'algorithm', 'interaction','time series analysis', 'stability','dynamical system']
# delete cross-domain terms
crosDom_15 = []


# "structure storage"
sorted_by_value = sorted(dict14.items(), key=lambda kv: kv[1], reverse= True)
keywords100_14 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_14 = ['com','reliability','availability','automation','age',
          'performance evaluation','analogy','functionality','raid',
          'object orient','scientific computation','component',
          'generator','workflow','software','desktop','compression',
          'proactive','log information','write performance',
          'energy conservation','response time']
# delete cross-domain terms
crosDom_14 = ['wiki','window','medical information',
              'geochronology','frequent pattern mining',
              'time series','telemedicine','teleophthalmology',
              'geodynamic','mid ocean ridge','parallel computation',
              'gps','context aware', 'web proxy server', 'feature extraction',
              'pattern recognition']


# "software engineer"
sorted_by_value = sorted(dict13.items(), key=lambda kv: kv[1], reverse= True)
keywords100_13 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_13 = ['survey','case study','software','test','communication','verification',
         'good practice', 'profile','agile','empirical study','metaheuristic', 
         'change impact analysis','formal method','energy consumption','interoperability',
         'cite','theory','challenge', 'requirement','management', 'static analysis',
         'energy efficiency', 'optimization','prototype','model','human factors']
# delete cross-domain terms
crosDom_13 = ['ontology','knowledge management', 'machine learn','genetic algorithm',
              'artificial intelligence','genetic programming','java','gamification',
              'bibliometric','uml','computer science','information retrieval','regression testing',
              'javascript', 'fuzzy logic','distributed system', 'radial basis function network',
              'taxonomy','bayesian network']


# "relational database"
sorted_by_value = sorted(dict12.items(), key=lambda kv: kv[1], reverse= True)
keywords100_12 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_12 = ['big datum','big data', 'security','performance','scalability',
          'robustness','benchmark','performance evaluation','graph',
          'architecture','condition','visualization','interoperability',
          '', 'model',]
# delete cross-domain terms
crosDom_12 = ['semantic web','semantic','datum mining','fuzzy logic','data mining',
              'cloud compute','random forest','genetic algorithm','knowledge discovery',
              'business intelligence','information retrieval','cluster',
              'geographic information system','context awareness','intrusion detection',
              'anomaly detection','cultural heritage','nuclear data mining',
              'machine learning','linguistic summary','graph mining','machine learn',
              'parallel compute']


# "parallel compute"
sorted_by_value = sorted(dict11.items(), key=lambda kv: kv[1], reverse= True)
keywords100_11 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_11 = ['big datum','cluster','optimization','benchmark','reliability',
          'complex network', 'energy efficiency','model','dynamic simulation',
          'simulation','contingency analysis']
# delete cross-domain terms
crosDom_11 = ['cloud computing','convolutional neural network', 
              'neural network', 'genetic algorithm','deep learn','datum mining',
              'community detection','heterogeneous computing','image segmentation',
              'bayesian', 'semi supervised learning','quantum computing',
              'k mean', 'nonlinear regression','marine electromagnetic geophysic']



# "operate system"
sorted_by_value = sorted(dict10.items(), key=lambda kv: kv[1], reverse= True)
keywords100_10 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_10 = ['performance','security','reliability','software','scalability',
          'performance evaluation','real time','static analysis','',
         'design', 'simulation','energy consumption','energy efficiency',
         'accessibility', 'vulnerability', 'policy','measurement','web',
         'privacy','fairness','attack','capability','education','internet',
         'tool','latency',
         ]
# delete cross-domain terms
crosDom_10 = ['cloud compute','cloud computing','data management','information security',
              'signal processing','augment reality','resource management','partition',
              'encryption','java','wireless sensor networks','hide markov model',
              'computer security','python', 'wi fi', 'private cloud']


# "machine learn"
sorted_by_value = sorted(dict9.items(), key=lambda kv: kv[1], reverse= True)
keywords100_9 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_9 = ['method', '','model','big datum','evaluation']
# delete cross-domain terms
crosDom_9 = ['image analysis','sentiment analysis','natural language processing',
             'image segmentation','pattern recognition', 'epilepsy',"alzheimer's disease",
             'heart rate', 'diagnostic','genetic algorithm','graph theory',"alzheimer's disease ad",
             'computer aided diagnosis','medical image','diagnosis','schizophrenia',
             'image classification','depression']


# "image processing"
sorted_by_value = sorted(dict8.items(), key=lambda kv: kv[1], reverse= True)
keywords100_8 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_8 = ['','technique','computational complexity', 'model','method',
         'algorithms','interoperability','robustness','dry','discriminant analysis']
# delete cross-domain terms
crosDom_8 = ['support vector machine','feature extraction','cluster','segmentation',
             'segmentation', 'cloud computing','classification','cancer',
             'biomedical measurement','osteoporosis','sparse representation',
             'feature representation', 'breast cancer','feature measurement',
             'dimensionality reduction','approximate computing','principal component analysis']


# "distribute compute"
sorted_by_value = sorted(dict7.items(), key=lambda kv: kv[1], reverse= True)
keywords100_7 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_7 = ['big datum', 'optimization','privacy','security','high performance computing', 
         'spark','performance','big data','fault tolerance','heuristic','energy efficiency',
         'simulation','benchmark','collaboration','comparison','consensus',
         'availability','scalability','reproducibility','analytic',
         'reliability','heterogeneity']
# delete cross-domain terms
crosDom_7 = ['wireless sensor network','machine learn','internet thing',
             'genetic algorithm','information security','workflow',
             'graph algorithm','resource management','sensor network','xml',
             'cluster','online social network','neural network']


# "data structure"
sorted_by_value = sorted(dict6.items(), key=lambda kv: kv[1], reverse= True)
keywords100_6 = [i[0] for i in sorted_by_value[:100]]

# delete general terms 
gen_6 = ['performance', 'algorithms','','language','big datum',
         'reliability','visualization','software','real time',
         'benchmark','safety','interoperability','design',
         'complexity','high performance computing','simulation',
         'global optimization','static analysis','algorithm',
         'analysis algorithms','schedule']

# delete cross-domain terms
crosDom_6 = ['dimensionality reduction','gpu computing','cloud computing',
             'cloud compute', 'fault detection','parallel algorithm',
             'graph algorithm','network', 'json','slide window','artificial intelligence',
             'program language', 'fuzzy cluster','parallel programming','semantic',
             'cloud security','r']


# "cryptography"
sorted_by_value = sorted(dict5.items(), key=lambda kv: kv[1], reverse= True)
keywords100_5 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_5 = ['security','privacy','s','','computational complexity','low latency',
         'lossless']
# delete cross-domain terms
crosDom_5 = ['wireless sensor network','mobile device','cloud compute',
             'entropy','cloud storage']


# "computer vision"
sorted_by_value = sorted(dict4.items(), key=lambda kv: kv[1], reverse= True)
keywords100_4 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_4 = ['survey','evaluation','energy efficiency']
# delete cross-domain terms
crosDom_4 = ['deep learn','feature extraction','convolutional neural network',
             'machine learn', 'support vector machine','classification',
             'sparse representation','deep neural network','recurrent neural network', 
             'neural network','random forest','feature selection','regression',
             'cluster','principal component analysis','cnn','gaussian process regression',
             'convolution','gaussian distribution','supervise learn','artificial neural network',
             'gaussian mixture model','approximate nearest neighbor','bayesian inference' ]


# "computer programming"
sorted_by_value = sorted(dict3.items(), key=lambda kv: kv[1], reverse= True)
keywords100_3 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_3 = ['learn','program','education','problem solve','motivation',
         'simulation','high education','assessment','mooc','teach',
         '','practice','model','student','elementary education',
         'teacher','arduino', 'technology', 'poetry','edit',
         'software','component','learn analytic','literacy', 
         'gender','human factors','expert find','algorithms',
         'experiment','ethic' ]
# delete cross-domain terms
crosDom_3 = ['game based learning', 'educational game','game',
             'game development','data mining','bioinformatic',
             'ontology','text mining','cluster']


# "computer graphics"
sorted_by_value = sorted(dict2.items(), key=lambda kv: kv[1], reverse= True)
keywords100_2 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_2 = ['', 'model','simulation','visibility','method','optimization',
         'approximation','education','design','analysis','smoothness',
         'product','innovative design']
# delete cross-domain terms
crosDom_2 = ['artificial intelligence','parallel computing','distribute compute',
             'k mean','unsupervised clustering', 'point cloud data']


# "bioinformatic"
sorted_by_value = sorted(dict1.items(), key=lambda kv: kv[1], reverse= True)
keywords100_1 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_1 = ['','pig', 'target','age','screen']
# delete cross-domain terms
crosDom_1 = ['feature selection']



# "algorithm design"
sorted_by_value = sorted(dict0.items(), key=lambda kv: kv[1], reverse= True)
keywords100_0 = [i[0] for i in sorted_by_value[:100]]
# delete general terms 
gen_0 = ['algorithm','optimization','','validation','scientific computing',
         'schedule','simulation', 'local search','visualization',
         'big data','optimisation','energy efficiency', 'heuristic algorithm',
         'uncertainty','satisfiability','configuration','security',
         'vulnerability'] #18
# delete cross-domain terms
crosDom_0 = ['wireless sensor network','neural network','cluster',
             'duplex radio','human compter interaction','computer vision',
             'web service','social network','lte','decode'] #10


############################ give the rest list as core labels

list(set(keywords100_0) - set(gen_0) - set(crosDom_0)) # difference between list | len:72 is correct

core_lables = []
for nb_label in range(17):
    keywords100 = globals()['keywords100_%s' % nb_label] 
    gen = globals()['gen_%s' % nb_label] 
    crosDom = globals()['crosDom_%s' % nb_label] 
    core_lables.append(list(set(keywords100) - set(gen) - set(crosDom)))


############################ merge domain from 17 to 11

# ' Bioinformatics  ',
# ' Cryptography  ',
# ' Machine learning  '
# ' Operating systems  ',
# ' Software engineering  ',
# ' network security  '
# ' Algorithm design  ', 
# ' Computer programming  '
             
# ' Distributed computing  ', ' Parallel computing  ',' Symbolic computation  ' | 7,11,15|--> 7
# ' Data structures  ', ' Structured Storage  ',  ' Relational databases  ' | 6, 12, 14|--> 6
# ' Computer graphics  ', ' Computer vision  ', ' Image processing  ', | 2, 4, 8 | --> 2

#(1). merge them
core_lables2 = []           
core_lables2 = core_lables

core_lables2[2].extend(core_lables2[4])
core_lables2[2].extend(core_lables2[8])
core_lables2[4]=[]
core_lables2[8]=[]

core_lables2[6].extend(core_lables2[12])
core_lables2[6].extend(core_lables2[14])
core_lables2[12]=[]
core_lables2[14]=[]

core_lables2[7].extend(core_lables2[11])
core_lables2[7].extend(core_lables2[15])
core_lables2[11]=[]
core_lables2[15]=[]


#(2). find the intersected terms in lists and exclude them 
core_lables3 = core_lables2

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

for i in range(17):
    for j in range(i+1,17): #only match once, not include itself
        intersec_res = intersection(core_lables3[i],core_lables3[j])
        if intersec_res != []:
            print(intersec_res, i, j)
            for item in intersec_res:
                core_lables3[i].remove(item)


#(3).  verify all give length as "1226"
counter=0
for i in core_lables:
    for j in i:
        counter +=1 #1226

counter=0
for i in core_lables2:
    for j in i:
        counter +=1 #1226
        
counter=0
for i in core_lables3:
    for j in i:
        counter +=1 #1142
                     
np.save(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3',core_lables3 )
#core_lables3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3'+'.npy')


##########!!! Manually evaluate the central terms in each subdomain ###########

######################## save them into excel for convinience
workbook = xlsxwriter.Workbook(os.getcwd()+'/intermediateRes/HeadRep_res/freqKeywordsSubdoamin_core_lables3.xlsx')
worksheet = workbook.add_worksheet()

# write header for each column
for i in range(len(core_lables3)):
    n_col_lines = i * 3 + 1
    worksheet.write(0, n_col_lines,i)# write name of column in first line
    print(i)
    
    n_row_lines = 1 # begin to write at the second line

    for k in core_lables3[i]:
        n_row_lines += 1
        worksheet.write(n_row_lines, n_col_lines,k)
        print(n_row_lines, n_col_lines,k)
    
workbook.close()



########################### re-load data without duplicates

# problem: it is really complex to transform the dataframe of Pandas to list 
#core_lables3_human = pd.read_excel('/home/ziwei/Desktop/KEScode/result_head_replacement/freqKeywordsSubdoamin_core_lables3.xlsx',
#                                   index_col=1)
# delete NaN value because of deletion of wrong value      
#core_lables3_human = core_lables3_human.dropna(how='all',axis=1) #delete enmpty column  , problem: ignore the fist column


from xlrd import open_workbook

book = open_workbook(os.getcwd()+'/intermediateRes/HeadRep_res/freqKeywordsSubdoamin_core_lables3_human.xlsx')
sheet = book.sheet_by_index(0) #If your data is on sheet 1

core_lables3_human = []
for col in range(50):
    
    core_lables3_human_t =[]
    for row in range(2, 218): #start from 1, to leave out row 0
        core_lables3_human_t.append(str(sheet.cell(row, col)))
        
    core_lables3_human.append(core_lables3_human_t)

# use regular expression to extract keywords
#re.match("text:'(.*)'", "text:'neighbor discovery'")
core_lables3_human2=[]
for subdomain in core_lables3_human:
    
    temp2 = []
    for item in subdomain:
        temp = re.match("text:'(.*)'", item)       
        if temp != None:
            print(temp)
            temp2.append(temp[1])
    core_lables3_human2.append(temp2)   

# to delete null list from the null column in excel
core_lables3_human3 = []
for subdomain in core_lables3_human2:
    if subdomain != []:
        core_lables3_human3.append(subdomain)

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human3',core_lables3_human3 )
#core_lables3_human3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human3'+'.npy')

# delete duplicated term
core_lables3_human4 = []
for subdomain in core_lables3_human3:
    temp = list(np.unique(subdomain))
    core_lables3_human4.append(temp)


np.save(os.getcwd()+'/intermediateRes/HeadRep_res/core_lables3_human4',core_lables3_human4 )
#core_lables3_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_recore_lables3_human4s/core_lables3_human4'+'.npy')




############!!! find hyponyms of central terms (fuzzy function) ##############       

#fuzz.ratio("ontology domain","ontology")
fuzz.partial_ratio("ontology domain","ontology")
#fuzz.token_sort_ratio("ontology domain","ontology") #regardless of sequency of terms
#fuzz.token_set_ratio("ontology domain","ontology") #whole tokens are included 


######################### ---v1---- find hyponyms directly from automatical central terms,
#########################           central terms without human evaluation 
# input: | NPs_PER_FILE_LIST | core_lables3 | 

NPs_PER_FILE_LIST = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')

lables_centralTerms_hyponyms = []
counter = 0
for inx_lables,lables in enumerate(core_lables3):
    
    lables_centralTerms_hyponyms_t = []
    for inx_centralTerms, centralTerms in enumerate(lables):
        
        # not single terms with letter less than 3
        if len(centralTerms) in [0,1,2]: 
            lables_centralTerms_hyponyms_t.append([])
        elif len(centralTerms) not in [0,1,2]: 
            counter += 1
            print(centralTerms,counter)
            
            centralTerms_hyponyms = []
            for file in NPs_PER_FILE_LIST:
                for item in file:
                    centralterms= core_lables3[inx_lables][inx_centralTerms]
                    if len(item.split())> len(centralterms.split()):
                        if fuzz.token_set_ratio(item,centralterms) == 100 : #bettern than <partial_ratio>, in case of ('script','operable description') 
                            centralTerms_hyponyms.append(item)
            lables_centralTerms_hyponyms_t.append(centralTerms_hyponyms)
        
    lables_centralTerms_hyponyms.append(lables_centralTerms_hyponyms_t)


np.save(os.getcwd()+'/intermediateRes/HeadRep_res/lables_centralTerms_hyponyms',lables_centralTerms_hyponyms )
#lables_centralTerms_hyponyms = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/lables_centralTerms_hyponyms'+'.npy')
# | 1142 | 16290 (unique:12347) |



######################### ---v2---- rebuild hyponyms from human evaluated central terms,
#########################           central terms with human evaluation 
# input: | NPs_PER_FILE_LIST | core_lables3_human4 | 

lables_centralTerms_hyponyms_human = []
counter = 0
for inx_lables,lables in enumerate(core_lables3_human4):
    
    lables_centralTerms_hyponyms_t = []
    for inx_centralTerms, centralTerms in enumerate(lables):
        
        if len(centralTerms) in [0,1,2]: 
            lables_centralTerms_hyponyms_t.append([])
        elif len(centralTerms) not in [0,1,2]: 
            counter += 1
            print(centralTerms,counter)
            
            centralTerms_hyponyms = []
            for file in NPs_PER_FILE_LIST:
                for item in file:
                    centralterms= core_lables3_human4[inx_lables][inx_centralTerms]
                    if len(item.split())> len(centralterms.split()):
                        if fuzz.token_set_ratio(item,centralterms) == 100 : #bettern than <partial_ratio>, in case of ('script','operable description') 
                            centralTerms_hyponyms.append(item)
            lables_centralTerms_hyponyms_t.append(centralTerms_hyponyms)
        
    lables_centralTerms_hyponyms_human.append(lables_centralTerms_hyponyms_t)

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/lables_centralTerms_hyponyms_human',lables_centralTerms_hyponyms_human )
#lables_centralTerms_hyponyms_human = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/lables_centralTerms_hyponyms_human'+'.npy')
# | 791 | 5507 (unique: 4298) | after human evaluate central terms


###------------------to evaluate the hypernym/hyponym pairs-----------------

########################## write into excel for hypernyms and their hyponyms
# l2_frequency(lables_centralTerms_hyponyms_human) #791
# l2_frequency(core_lables3_human4) # 791

workbook = xlsxwriter.Workbook(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym.xlsx')
worksheet = workbook.add_worksheet()
###### structure
###### 1. column 0: label + hypernyms
###### 2. column 1 -> later: label + hyponyms

# write hypernyms in first column | core_lables3_human4 |
counter =0
for i in range(len(core_lables3_human4)):
    counter += 1
    
    worksheet.write(counter, 0,i) # give the separation for each domain
    
    for item in core_lables3_human4[i]:
        counter += 1
        worksheet.write(counter,0,item)
# write hyponyms in second column | lables_centralTerms_hyponyms_human |
counter =0
for i in range(len(lables_centralTerms_hyponyms_human)):
    counter += 1
    
    worksheet.write(counter, 1,i) # give the separation for each domain
    
    for item in lables_centralTerms_hyponyms_human[i]:
        counter += 1
        for j in range(len(item)):
            worksheet.write(counter,1+j,item[j])   
workbook.close()

################################  human evluatioan takes 4-6 hours


################################ read the human evaluated excel 
###  | hypernym_1_N_hyponym_human |

from xlrd import open_workbook

book = open_workbook(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human.xlsx')
sheet = book.sheet_by_index(0) #If your data is on sheet 1
# first column is the same as | core_lables3_human4 |
# >=second column is the human evaluated content

hypernym_1_N_hyponym_human = []
counter =0
for i in range(len(lables_centralTerms_hyponyms_human)):
    counter += 1
    
    hypernym_1_N_hyponym_human_t =[]
    
    for item in lables_centralTerms_hyponyms_human[i]:
        counter += 1
#        print(i,counter, len(item))
        
        hypernym_1_N_hyponym_human_tt=[]
        if len(item) <265: # error, i cannot find | 449 in counter = 290
            for j in range(len(item)):
                hypernym_1_N_hyponym_human_tt.append(str(sheet.cell(counter,1+j)))
        else:
            hypernym_1_N_hyponym_human_tt.append("empty:''")
            
        hypernym_1_N_hyponym_human_t.append(hypernym_1_N_hyponym_human_tt)

    hypernym_1_N_hyponym_human.append(hypernym_1_N_hyponym_human_t)
    
np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human',hypernym_1_N_hyponym_human )
#hypernym_1_N_hyponym_human = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human'+'.npy')


# (1). extrac with only value with pattern 
    
# use regular expression to extract keywords
#re.match("text:'(.*)'", "text:'neighbor discovery'")
hypernym_1_N_hyponym_human2=[]
for subdomain in hypernym_1_N_hyponym_human:
    
    temp2 = []
    for item in subdomain:
        temp3 = []
        for words in item: 
            temp = re.match("text:'(.*)'", words)  
            if temp != None:
#             print(temp)
                temp3.append(temp[1])
        temp2.append(temp3)
    hypernym_1_N_hyponym_human2.append(temp2) 
    
np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human2',hypernym_1_N_hyponym_human2 )
#hypernym_1_N_hyponym_human2 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human2'+'.npy')


# (2). delete duplicated term
hypernym_1_N_hyponym_human3 = []

for subdomain in hypernym_1_N_hyponym_human2:
    temp1=[]
    for term in subdomain:
        temp = list(np.unique(term))
        temp1.append(temp)
    hypernym_1_N_hyponym_human3.append(temp1)

#update: 16th April 2019 # add subdomain name into Gold Standard(hypernym_1_N_hyponym_human3)
#  name_subdomain11
for i in range(11):
    hypernym_1_N_hyponym_human3[i].append([name_subdomain11[i]])

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human3',hypernym_1_N_hyponym_human3 )
#hypernym_1_N_hyponym_human3 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human3'+'.npy')
# | 1951 | unique = 1951



# !!! ------------------replace with central terms in file(partial)------------
# input: | NPs_PER_FILE_LIST | : pure NPs in file
#        | lables_centralTerms_hyponyms | : central terms in each label file
#        | nounlistinFile2 | : NPs with attributes in file

# output: | hyponym_hypernym_subdomain_list | : <hyponyms, #lables, #indexOfTerms, hypernyms>
#         | Nps_rep_PerfileList | : pure NPs with replaced by hypernyms in file
#         | nounlistinFile2_headRep_v2 | : NPs(replaced) with attributes in file



def replacedByCentralTerms(NPs_PER_FILE_LIST, lables_centralTerms_hyponyms, 
                           core_lables3, nounlistinFile2):

    hyponym_hypernym_subdomain_list = []
    Nps_rep_PerfileList = NPs_PER_FILE_LIST
    nounlistinFile2_headRep_v2 = [] # with verb information
    counter = 0
    
    for inx_file, file in enumerate(NPs_PER_FILE_LIST):
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




hyponym_hypernym_subdomain_list, Nps_rep_PerfileList, nounlistinFile2_headRep_v2 = replacedByCentralTerms(NpsPerfileList, lables_centralTerms_hyponyms, core_lables3, nounlistinFile2)
# | 16284 (original: 16290) times to be replaced | with automatically hypernyms and hyponyms
# JUly 21th Update: 1739|344| 332 times to be replaced ( the results are not stable)

# change to | lables_centralTerms_hyponyms_human  (unique:from 12347 to 4298)| and | core_lables3_human4 
hyponym_hypernym_subdomain_list, Nps_rep_PerfileList, nounlistinFile2_headRep_v2 = replacedByCentralTerms(NpsPerfileList, lables_centralTerms_hyponyms_human, core_lables3_human4, nounlistinFile2)
# JUly 21th Update: 70 times to be replaced ( the results are not stable)


np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hyponym_hypernym_subdomain_list',hyponym_hypernym_subdomain_list )
#hyponym_hypernym_subdomain_list = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hyponym_hypernym_subdomain_list'+'.npy')

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/Nps_rep_PerfileList2',Nps_rep_PerfileList )
#Nps_rep_PerfileList = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/Nps_rep_PerfileList2'+'.npy')

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/nounlistinFile2_headRep_v2',nounlistinFile2_headRep_v2 )
#nounlistinFile2_headRep_v2 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/nounlistinFile2_headRep_v2'+'.npy')



# input: | NpsPerfileList | : pure NPs in file
#        | hypernym_1_N_hyponym_human3 | : hyponyms of central terms 
#        | nounlistinFile2 | : NPs with attributes in file
#        | core_lables3_human4 |: the central terms

# output: | hyponym_hypernym_subdomain_list_human | : <hyponyms, #lables, #indexOfTerms, hyponyms>
#         | Nps_rep_PerfileList_human | : pure NPs with replaced by hypernyms in file
#         | nounlistinFile2_headRep_v2_human | : NPs(replaced) with attributes in file


TODO: to check the different between variables and find the possibility to use the function or modificatiton 
l2_frequency(hypernym_1_N_hyponym_human3)
l2_frequency(core_lables3_human4)


hyponym_hypernym_subdomain_list_human, Nps_rep_PerfileList_human, nounlistinFile2_headRep_v2_human = replacedByCentralTerms(NpsPerfileList, hypernym_1_N_hyponym_human3, core_lables3_human4, nounlistinFile2)
hyponym_hypernym_subdomain_list, Nps_rep_PerfileList, nounlistinFile2_headRep_v2 = replacedByCentralTerms(NpsPerfileList, lables_centralTerms_hyponyms, core_lables3, nounlistinFile2)
# problem: each list of hypernym_1_N_hyponym_human3 has one more element. 
# solution: 
# problem:  hyponym_hypernym_subdomain_list_human is empty


test1 = l2_plain(NpsPerfileList)
test2 = l3_plain(lables_centralTerms_hyponyms)
np.intersect1d(np.array(test1), np.array(test2)).size #67

test1 = l2_plain(NpsPerfileList)
test2 = l3_plain(hypernym_1_N_hyponym_human3)
np.intersect1d(np.array(test1), np.array(test2)).size #1
# problem: the string matching rate is really low, we can use fuzzywuzzy string matching


#-------------------------------v1--------------------------
test1 = l2_plain(NpsPerfileList) #291,488
test2 = l3_plain(hypernym_1_N_hyponym_human3) # 1951

%%time
counter = 0 
for i1 in test1:
    if len(i1.split()) >= 3:  # in case that : propose method| hyponyms:propose saliency detection method ;;;; volume| hyponyms:volume finite element method
        for i2 in test2:
            if fuzz.token_set_ratio(i1, i2) == 100:
                counter += 1
                print("originals:"+i1+"| hyponyms:"+i2)
                print(counter)

# the idea to find the replaced coreterms or domains for NP1
# e.g. 
'''
originals:GA LM BP neural network model| hyponyms:bp neural network

originals:GA LM BP neural network model| hyponyms:bp neural network model

originals:GA LM BP neural network model| hyponyms:ga lm bp neural network model

originals:GA LM BP neural network model| hyponyms:neural network model
'''

test4 = ["bp neural network","bp neural network model","ga lm bp neural network model","neural network model"]
temp4= []
for inx_file, file in enumerate(hypernym_1_N_hyponym_human3):
    for inx_central, central in enumerate(file):
        for i in test4: 
            if i in central:
#                print(i+" | " + core_lables3_human4[inx_file][inx_central])
                temp4.append((i, core_lables3_human4[inx_file][inx_central]))


# the results show that they are the same central terms. If they are not the same central terms, we will choose by frequency
#-------------------------------v2--------------------------
test1 = l2_plain(NpsPerfileList) #291,488
test2 = l3_plain(hypernym_1_N_hyponym_human3) # 1951

%%time
counter = 0 
for i1 in test1:
    if len(i1.split()) >= 3:  # in case that : propose method| hyponyms:propose saliency detection method ;;;; volume| hyponyms:volume finite element method
        for i2 in test2:
            if fuzz.token_set_ratio(i1, i2) == 100:
                counter += 1
                print("originals:"+i1+"| hyponyms:"+i2)
                print(counter)             


                
# diffirence between v1 and v2 function:
#   1. add new input variable : name_subdomain
#   2. counter
#   3. find the roughly matched string with fuzzywuzzy, rather than <string in list>
                
def replacedByCentralTerms5(NPs_PER_FILE_LIST, lables_centralTerms_hyponyms, 
                           core_lables3, nounlistinFile2, name_subdomain):

    hyponym_hypernym_subdomain_list = []
    Nps_rep_PerfileList = copy.deepcopy(NPs_PER_FILE_LIST)
    nounlistinFile2_headRep_v2 = [] # with verb information
    counter = 0
    
    for inx_file, file in enumerate(NPs_PER_FILE_LIST):
        print("fileID"+str(inx_file))
        nounlistinFile2_headRep_v2_t = []
        hyponym_hypernym_subdomain_list_t = []
        for inx_item, item in enumerate(file):
            searchTerm = item
            term_2 = nounlistinFile2[inx_file][inx_item]
            
            # to test whether <searchTerm> exist in <lables_centralTerms_hyponyms>
            temp = []  
            temp4 = [] # store all the roughly matched hypos for a searchTerms
            for inx_i, i in enumerate(lables_centralTerms_hyponyms):
                for inx_j, j in enumerate(i):
                    if len(searchTerm.split()) >= 3:
                        for hypo in j:
                            if fuzz.token_set_ratio(searchTerm, hypo) == 100:
#                                counter += 1
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
                    counter += 1
    
                elif len(temp)==1: 
                    
                    # store in list
                    if temp[0][0] != temp[0][3]:
                        hyponym_hypernym_subdomain_list_t.append(temp[0]) 
                    # np with verb information 
                    nounlistinFile2_headRep_v2_t.append((term_2[0],temp[0][3], term_2[2],term_2[3],term_2[4]))  
                    counter += 1                       
            
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

# too slow to get the results (there are so many elements in lables_centralTerms_hyponyms)
#hyponym_hypernym_subdomain_list, Nps_rep_PerfileList, nounlistinFile2_headRep_v2 = replacedByCentralTerms2(NpsPerfileList, lables_centralTerms_hyponyms, core_lables3, nounlistinFile2,name_subdomain)

# to refresh, in case that the original datasets are overwrited along with variables
NpsPerfileList = np.load(os.getcwd()+'/intermediateRes/NpsPerfileList'+'.npy')
nounlistinFile2 = np.load(os.getcwd()+'/intermediateRes/nounlistinFile2'+'.npy')
hyponym_hypernym_subdomain_list_human, Nps_rep_PerfileList_human, nounlistinFile2_headRep_v2_human = replacedByCentralTerms5(NpsPerfileList, hypernym_1_N_hyponym_human3, core_lables3_human4, nounlistinFile2,name_subdomain11)
# len(l2_plain(hyponym_hypernym_subdomain_list_human)) # 4642 times to be subsituted 
# len(np.unique(l2_plain(hyponym_hypernym_subdomain_list_human))) #3248 unique terms are substituted


np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hyponym_hypernym_subdomain_list_human',hyponym_hypernym_subdomain_list_human )
#hyponym_hypernym_subdomain_list_human = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hyponym_hypernym_subdomain_list_human'+'.npy')

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/Nps_rep_PerfileList_human',Nps_rep_PerfileList_human )
#Nps_rep_PerfileList_human = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/Nps_rep_PerfileList_human'+'.npy')

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/nounlistinFile2_headRep_v2_human',nounlistinFile2_headRep_v2_human )
#nounlistinFile2_headRep_v2_human = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/nounlistinFile2_headRep_v2_human'+'.npy')





# enlarge GS (hypernym_1_N_hyponym_human3) with new extracted NPs

# to find the range of domain ID
DomainID = list(np.unique([j[1] for i in hyponym_hypernym_subdomain_list_human for j in i])) # results: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
plain_hyponym_hypernym_subdomain_list_human = [j for i in hyponym_hypernym_subdomain_list_human for j in i]

temp3 = []
for domainid in DomainID: 
    centralTermsId = np.unique([i[2] for i in plain_hyponym_hypernym_subdomain_list_human if i[1] == domainid])
    temp2 = []
    for centraltermid in range(max(centralTermsId)):
        temp1 = []
        for i in plain_hyponym_hypernym_subdomain_list_human:
            if i[1] == domainid and  i[2] == centraltermid :
                temp1.append(i[0])  
        # to delete the duplicate terms as hyponyms of one central terms
        temp1 = list(np.unique(temp1))
        temp2.append(temp1)
    temp3.append(temp2)

hypernym_1_N_hyponym_human4 = temp3

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4',hypernym_1_N_hyponym_human4 )
#hypernym_1_N_hyponym_human4 = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4'+'.npy')


l3_frequency_u(hypernym_1_N_hyponym_human4) #2776 unique terms for this new GS
l3_frequency_u(hypernym_1_N_hyponym_human3) #1962 unique terms for this new GS
    



# divide the new GS into 2 parts (trainingGS 80% and testing GS 20%)

hypernym_1_N_hyponym_human4_train = copy.deepcopy(hypernym_1_N_hyponym_human4)
hypernym_1_N_hyponym_human4_test = copy.deepcopy(hypernym_1_N_hyponym_human4)

counter = 0
for inx_d, domains in enumerate(hypernym_1_N_hyponym_human4):
    for inx_c, centralTerms in enumerate(domains):
        for inx_h, hypo in enumerate(centralTerms):
            counter += 1
            if counter % 10 not in [8,9]: # stay in train set, remove from test set
                hypernym_1_N_hyponym_human4_test[inx_d][inx_c].remove(hypo)
            else: # stay in test set, remove from train set
                hypernym_1_N_hyponym_human4_train[inx_d][inx_c].remove(hypo)

                
l3_frequency(hypernym_1_N_hyponym_human4_test) #554
l3_frequency(hypernym_1_N_hyponym_human4_train) #2222

np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test',hypernym_1_N_hyponym_human4_test )
np.save(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train',hypernym_1_N_hyponym_human4_train )
# corresponding to <name_subdomain11_v2>

