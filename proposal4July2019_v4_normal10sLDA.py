#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:19:15 2019

    purpose: train only once LDA and evaluate the results
@author: ziwei
"""

############## case1 (10 times) ##########
termsPerfileList_case1 = [goldNPs_set_case1, goldNPs_set_rep_case1, goldNPs_set_keywords_case1, goldNPs_set_keywords_rep_case1]

filter_no_below = 0
filter_no_above = 1
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case1 = []
res_stas_case1 = []
for termsPerfileList in termsPerfileList_case1:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case1.append(res_avg)
    res_stas_case1.append(res_stas)
    
    
############## case2 (10 times) ##########
termsPerfileList_case2 = [goldNPs_set_case2, goldNPs_set_rep_case2, goldNPs_set_keywords_case2, goldNPs_set_keywords_rep_case2]

filter_no_below = 0
filter_no_above = 1
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case2 = []
res_stas_case2 = []
for termsPerfileList in termsPerfileList_case2:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case2.append(res_avg)
    res_stas_case2.append(res_stas)
    
    
    
############## case3 (10 times) ##########
termsPerfileList_case3 = [goldNPs_set_case3, goldNPs_set_rep_case3, goldNPs_set_keywords_case3, goldNPs_set_keywords_rep_case3]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case3 = []
res_stas_case3 = []
for termsPerfileList in termsPerfileList_case3:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case3.append(res_avg)
    res_stas_case3.append(res_stas)
    
    
############## case4 (10 times) ##########
termsPerfileList_case4 = [goldNPs_set_case4, goldNPs_set_rep_case4, goldNPs_set_keywords_case4, goldNPs_set_keywords_rep_case4]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case4 = []
res_stas_case4 = []
for termsPerfileList in termsPerfileList_case4:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case4.append(res_avg)
    res_stas_case4.append(res_stas)   
    
    
############## case5 (10 times) ##########
termsPerfileList_case5 = [goldNPs_set_case5, goldNPs_set_rep_case5, goldNPs_set_keywords_case5, goldNPs_set_keywords_rep_case5]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case5 = []
res_stas_case5 = []
for termsPerfileList in termsPerfileList_case5:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case5.append(res_avg)
    res_stas_case5.append(res_stas)   
    
    
############## case6 (10 times) ##########
termsPerfileList_case6 = [goldNPs_set_case6, goldNPs_set_rep_case6, goldNPs_set_keywords_case6, goldNPs_set_keywords_rep_case6]

filter_no_below = 5
filter_no_above = 0.5
tp = 50

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case6 = []
res_stas_case6 = []
for termsPerfileList in termsPerfileList_case6:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case6.append(res_avg)
    res_stas_case6.append(res_stas)   
    

np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case1',res_avg_case1)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case1',res_stas_case1)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case2',res_avg_case2)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case2',res_stas_case2)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case3',res_avg_case3)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case3',res_stas_case3)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case4',res_avg_case4)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case4',res_stas_case4)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case5',res_avg_case5)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case5',res_stas_case5)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case6',res_avg_case6)
np.save(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_stas_case6',res_stas_case6)

#np.load(os.getcwd()+'/LDAtrain_res/normal10sLDA_5_50p/res_avg_case1.npy')



### the base method to put these variables into latex (in case that we input with wrong parameters)

import scipy
from tabulate import tabulate
a = scipy.random.rand(3,3)

print(tabulate(a,tablefmt = "latex", floatfmt = ".2f"))







#test

############## case6 (10 times) ##########
termsPerfileList_case6 = [goldNPs_set_case6, goldNPs_set_rep_case6, goldNPs_set_keywords_case6, goldNPs_set_keywords_rep_case6]

filter_no_below = 5
filter_no_above = 0.5
tp = 100

# to nest tuples   (u= t, (1,2,3,4))
res_avg_case6 = []
res_stas_case6 = []
for termsPerfileList in termsPerfileList_case6:
    res_avg, res_stas = train_eval_10_v2(termsPerfileList,filter_no_below, filter_no_above, tp,threshold0_freq = 3000, threshold1_termProb = 1e-8, threshold2_termSign = 0.001)
    res_avg_case6.append(res_avg)
    res_stas_case6.append(res_stas)   




    