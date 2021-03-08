
import os
import numpy as np
import joblib 
import pandas as pd
from collections import defaultdict
# 对保存的trajectory进行分析，统计不同target出现的次数、成功的次数和每局的结果。
def analysis_trajectory(trajectory_dic, savepath, winReward=1):
    print("Start analysis the trajectories....")
    all_obvers = trajectory_dic["all_obvs"]
    all_dones = trajectory_dic["all_dones"].flatten()
    all_rewards = trajectory_dic["all_rwds"].flatten()
    ends_index = np.where(all_dones==True)[0]
    success_idx = ends_index[np.where(all_rewards[ends_index]==winReward)[0]]
    print("Total episodes:",ends_index.shape)
    print("Success episodes:",success_idx.shape)
    
    upCounts = np.array([0] * len(ends_index))
    downCounts = np.array([0] * len(ends_index))
    downTotal = np.array([0] * len(ends_index))
    upTotal = np.array([0] * len(ends_index))
    results = np.array([-1] * len(ends_index))
    for i,endIdx in enumerate(ends_index):
        # observation: [cur_x, cur_y, target_x, target_y]
        if all_obvers[endIdx][-1] != 0: # upTarget, downTotal and downCounts keep same as previous timestep.
            upTotal[i] = 1 if i==0 else upTotal[i-1] + 1
            downTotal[i] =  0 if i==0 else downTotal[i-1] 
            downCounts[i] = 0 if i==0 else downCounts[i-1]
            if all_rewards[endIdx] == winReward:
                results[i] = 1
                upCounts[i] = 1 if i==0 else upCounts[i-1] + 1
            else:
                results[i] = 0
                upCounts[i] = upCounts[i-1]

        elif all_obvers[endIdx][-1] == 0: # downTarget, upTotal and upCounts keep same as previous timestep.
            downTotal[i] = 1 if i==0 else downTotal[i-1] + 1
            upTotal[i] =  0 if i==0 else upTotal[i-1] 
            upCounts[i] = 0 if i==0 else upCounts[i-1]
            if all_rewards[endIdx] == winReward:
                results[i] = 1
                downCounts[i] = 1 if i==0 else downCounts[i-1] + 1
            else:
                results[i] = 0
                downCounts[i] = downCounts[i-1]
        else:
            assert False, "Error Targets!(obv:{})".format(all_obvers[endIdx])

    print("success for upTargets:{}/{}".format(upCounts[-1], upTotal[-1]))
    print("success for downTargets:{}/{}".format(downCounts[-1], downTotal[-1]))
    results_dic = {
        "ends_index":ends_index,
        "upCounts":upCounts,
        "upTotal":upTotal,
        "downCounts":downCounts,
        "downTotal":downTotal,
        "results":results
    }
    joblib.dump(results_dic,savepath+"_res.dic")
    print("Analysis results are saved in", savepath+"_res.dic")
    return results_dic


# 计算最近slideLen局的胜率情况，upRate = #(最近100局upTarget出现且成功的次数)/#(最近100局upTarget出现的次数)
def fetchRecentSR(results_dic, showEpisodes=None, slideLen = 100):
    ends_index = results_dic['ends_index']
    upCounts = results_dic['upCounts']
    upTotal = results_dic['upTotal']
    downCounts = results_dic['downCounts']                         
    downTotal = results_dic['downTotal']
    
    idx_list = [i for i in range(len(ends_index))]
    n_episodes = idx_list if showEpisodes is None else idx_list[:showEpisodes]
    
    upRates, downRates = [], []
    recentUp, recentDown = [], []
    for i in n_episodes:
        if upTotal[i] == 0:
            assert upCounts[i]==0
            upRates.append(0.)
            recentUp.append(0)
        else:
            if i<slideLen: 
                upRates.append(upCounts[i]/upTotal[i])
                recentUp.append(upTotal[i])
            else:  
                upRates.append((upCounts[i]-upCounts[i-slideLen])/(upTotal[i]-upTotal[i-slideLen]))
                recentUp.append(upTotal[i]-upTotal[i-slideLen])
        
        if downTotal[i] == 0:
            assert downCounts[i]==0
            downRates.append(0.)
            recentDown.append(0)
        else:
            if i<slideLen: 
                downRates.append(downCounts[i]/downTotal[i])
                recentDown.append(downTotal[i])
            else:  
                downRates.append((downCounts[i]-downCounts[i-slideLen])/(downTotal[i]-downTotal[i-slideLen]))
                recentDown.append(downTotal[i]-downTotal[i-slideLen])
    
    return upRates,downRates
    

# 返回不同种子下所有游戏的统计信息
def fetchProfileDF(method, policy, envName, timeSteps, seeds, save_dir, isRandom=False):
    # {'seed_1':[episode_num, successUp, totalUp, successDown, totalDown]}
    profile_dic = defaultdict(list)
    all_results =  defaultdict()
    for seed in seeds:
        saveanme = "{}_{}_{}_{}_{}_{}".format(method, policy, envName, int(isRandom), timeSteps, seed)
        savepath = os.path.join(save_dir,saveanme)
        results_dic = joblib.load(savepath+"_res.dic")
        
        episode_num = len(results_dic['ends_index'])
        successUp,totalUp =  results_dic['upCounts'][-1],results_dic['upTotal'][-1]
        successDown,totalDown =  results_dic['downCounts'][-1],results_dic['downTotal'][-1]
        profile_dic[seed] = [episode_num,successUp,totalUp,successDown,totalDown]
        all_results[seed] = results_dic

    profile_data = pd.DataFrame(profile_dic)
    profile_data.index = ['episode_num','successUp','totalUp','successDown','totalDown']
    xlimit = min(list(profile_data.loc['episode_num']))
    print("Average results of first {} games over seeds({}->{}) will be calculated.".format(xlimit,seeds[0],seeds[-1]))

    return xlimit, profile_dic, all_results

def create_new_df(old_df):
    df_list = []
    for seed in old_df.columns:
        cur_df = pd.DataFrame({'idx':old_df.index,'result':old_df[seed],'seed':[seed]*len(old_df[seed])})
        df_list.append(cur_df)
    return pd.concat(df_list)

def createPlotData(seeds, xlimit, all_results, showResult):

    # {'seed_1':[1,...,xlimit]}
    all_result_dic = defaultdict(list)

    for seed in seeds:
        results_dic = all_results[seed]
        upRates,downRates = fetchRecentSR(results_dic)
        if showResult == "downRate":
            all_result_dic[seed] = downRates[:xlimit]
        elif showResult == "upRate":
            all_result_dic[seed] = upRates[:xlimit]
        elif showResult == "result":
            all_result_dic[seed] = results_dic['results'][:xlimit]

    old_df = pd.DataFrame(all_result_dic)   

    return old_df