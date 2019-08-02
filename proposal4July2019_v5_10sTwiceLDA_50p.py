#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:13:41 2019
    twice trained LDA
    #1. after the first LDA training, use the function <clusterDistribution5> to get the evident terms from dictionary list
    #2. use the terms to re-constituted the new corpus in files with function <NPinFileSelection>
    #3. use the new re-constituted corpus to train 10 times LDA and get the average results
    
    TODO: results are bad,  try to level up the number of topics, so as to decrease the number of elements in each cluster
    
    
    WHAT WE FOUND HERE?
    Time: the time consuming for case5 and case6 is so high! 
    the average number for each cluster is 100, thus we have 5000 terms to be clustered at end.
    the average number for each cluster is 100, thus we have 7000 terms to be clustered at end.    
    
    
@author: ziwei
"""


# for evaluation: keep these GS terms from elimination from filtering
GS_train = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_train'+'.npy') 
GS_test = np.load(os.getcwd()+'/intermediateRes/HeadRep_res/hypernym_1_N_hyponym_human4_test'+'.npy') 
keepToken = l3_plain(GS_train)
keepToken.extend(l3_plain(GS_test))
#len(keepToken) #2776



goldNPs_set_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case3'+'.npy')
goldNPs_set_rep_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case3'+'.npy')
goldNPs_set_keywords_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case3'+'.npy')
goldNPs_set_keywords_rep_case3 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case3'+'.npy')

goldNPs_set_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case4'+'.npy')
goldNPs_set_rep_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case4'+'.npy')
goldNPs_set_keywords_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case4'+'.npy')
goldNPs_set_keywords_rep_case4 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case4'+'.npy')

goldNPs_set_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case5'+'.npy')
goldNPs_set_rep_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case5'+'.npy')
goldNPs_set_keywords_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case5'+'.npy')
goldNPs_set_keywords_rep_case5 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case5'+'.npy')

goldNPs_set_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_case6'+'.npy')
goldNPs_set_rep_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_rep_case6'+'.npy')
goldNPs_set_keywords_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_case6'+'.npy')
goldNPs_set_keywords_rep_case6 = np.load(os.getcwd()+'/intermediateRes/FourCases_res/goldNPs_set_keywords_rep_case6'+'.npy')





# ----------example of one input for twice LDA-------------------- #

#termsPerfileList =  goldNPs_set_case3      # l2_frequency(goldNPs_set_case3) #107737   
#termsPerfileList =  goldNPs_set_rep_case3
#termsPerfileList =  goldNPs_set_keywords_case3       
#termsPerfileList =  goldNPs_set_keywords_rep_case3
#
#filter_no_below = 5
#filter_no_above = 0.5
#tp = 50
#
#lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp,KEEP_TOKEN = keepToken) 
## len(NPsdictionary) #3413
############### train LDA and evaluate (10 times) (2nd stage) ##########
#
#reservedTerms1stLDA = clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
## len(reservedTerms1stLDA) # 3413
#
#case_DT_1stLDA =  NPinFileSelection(termsPerfileList, reservedTerms1stLDA)
## l2_frequency(case_DT_1stLDA) #49770
#
#res_avg_case1, res_stas_case1 = train_eval_10_v2(case_DT_1stLDA,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)





############## case3 (twice LDA) ##########
termsPerfileList_case3 = [goldNPs_set_case3, goldNPs_set_rep_case3, goldNPs_set_keywords_case3, goldNPs_set_keywords_rep_case3]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case3_twice = []
res_stas_case3_twice = []
for termsPerfileList in termsPerfileList_case3:  
    lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp,KEEP_TOKEN = keepToken) 
    reservedTerms1stLDA = clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    case_DT_1stLDA =  NPinFileSelection(termsPerfileList, reservedTerms1stLDA)
    res_avg, res_stas = train_eval_10_v2(case_DT_1stLDA,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

    res_avg_case3_twice.append(res_avg)
    res_stas_case3_twice.append(res_stas)
    

############## case4 (twice LDA) ##########
termsPerfileList_case4 = [goldNPs_set_case4, goldNPs_set_rep_case4, goldNPs_set_keywords_case4, goldNPs_set_keywords_rep_case4]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case4_twice = []
res_stas_case4_twice = []
for termsPerfileList in termsPerfileList_case4:  
    lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp,KEEP_TOKEN = keepToken) 
    reservedTerms1stLDA = clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    case_DT_1stLDA =  NPinFileSelection(termsPerfileList, reservedTerms1stLDA)
    res_avg, res_stas = train_eval_10_v2(case_DT_1stLDA,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

    res_avg_case4_twice.append(res_avg)
    res_stas_case4_twice.append(res_stas)




############## case5 (twice LDA) ##########
termsPerfileList_case5 = [goldNPs_set_case5, goldNPs_set_rep_case5, goldNPs_set_keywords_case5, goldNPs_set_keywords_rep_case5]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case5_twice = []
res_stas_case5_twice = []
for termsPerfileList in termsPerfileList_case5:  
    lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp,KEEP_TOKEN = keepToken) 
    reservedTerms1stLDA = clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    case_DT_1stLDA =  NPinFileSelection(termsPerfileList, reservedTerms1stLDA)
    res_avg, res_stas = train_eval_10_v2(case_DT_1stLDA,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

    res_avg_case5_twice.append(res_avg)
    res_stas_case5_twice.append(res_stas)


############## case6 (twice LDA) ##########
termsPerfileList_case6 = [goldNPs_set_case6, goldNPs_set_rep_case6, goldNPs_set_keywords_case6, goldNPs_set_keywords_rep_case6]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case6_twice = []
res_stas_case6_twice = []
for termsPerfileList in termsPerfileList_case6:  
    lda_model, processed_time, NPsdictionary, bow_corpus, corpus_tfidf = trainLDAModel(termsPerfileList, filter_no_below, filter_no_above,NUM_topics=tp,KEEP_TOKEN = keepToken) 
    reservedTerms1stLDA = clusterDistribution5(bow_corpus, NPsdictionary, lda_model,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    case_DT_1stLDA =  NPinFileSelection(termsPerfileList, reservedTerms1stLDA)
    res_avg, res_stas = train_eval_10_v2(case_DT_1stLDA,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)

    res_avg_case6_twice.append(res_avg)
    res_stas_case6_twice.append(res_stas)





np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_avg_case3_twice',res_avg_case3_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_stas_case3_twice',res_stas_case3_twice)

np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_avg_case4_twice',res_avg_case4_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_stas_case4_twice',res_stas_case4_twice)

np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_avg_case5_twice',res_avg_case5_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_stas_case5_twice',res_stas_case5_twice)

np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_avg_case6_twice',res_avg_case6_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_50p/res_stas_case6_twice',res_stas_case6_twice)




