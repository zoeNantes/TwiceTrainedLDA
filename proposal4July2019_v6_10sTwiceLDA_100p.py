#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:25:57 2019

@author: ziwei
"""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:13:41 2019

    Difference from <proposal4July2019_v5_10sTwiceLDA_50p.py>:
        #1. level up the number of topics from 50 to 100. and compare the results
      
    
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






############## case3 (twice LDA) ##########
termsPerfileList_case3 = [goldNPs_set_case3, goldNPs_set_rep_case3, goldNPs_set_keywords_case3, goldNPs_set_keywords_rep_case3]

filter_no_below = 5
filter_no_above = 0.5
tp = 100

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
    
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_avg_case3_twice',res_avg_case3_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_stas_case3_twice',res_stas_case3_twice)

############## case4 (twice LDA) ##########
termsPerfileList_case4 = [goldNPs_set_case4, goldNPs_set_rep_case4, goldNPs_set_keywords_case4, goldNPs_set_keywords_rep_case4]

filter_no_below = 5
filter_no_above = 0.5
tp = 100

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

np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_avg_case4_twice',res_avg_case4_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_stas_case4_twice',res_stas_case4_twice)


############## case5 (twice LDA) ##########
termsPerfileList_case5 = [goldNPs_set_case5, goldNPs_set_rep_case5, goldNPs_set_keywords_case5, goldNPs_set_keywords_rep_case5]

filter_no_below = 5
filter_no_above = 0.5
tp = 100

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


np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_avg_case5_twice',res_avg_case5_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_stas_case5_twice',res_stas_case5_twice)



############## case6 (twice LDA) ##########
termsPerfileList_case6 = [goldNPs_set_case6, goldNPs_set_rep_case6, goldNPs_set_keywords_case6, goldNPs_set_keywords_rep_case6]

filter_no_below = 5
filter_no_above = 0.5
tp = 100

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


np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_avg_case6_twice',res_avg_case6_twice)
np.save(os.getcwd()+'/LDAtrain_res/twice10sLDA_5_100p/res_stas_case6_twice',res_stas_case6_twice)




