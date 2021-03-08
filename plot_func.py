

import numpy as np
from utils import fetchRecentSR, fetchProfileDF
from utils import createPlotData, create_new_df
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

def plot_all_results(envName, save_dir,seeds,results, slideLen = 100):

    fig = plt.figure(figsize=(24,10))
    fig,axs = plt.subplots(nrows=2, ncols=5,figsize=(24,8))
    for i,seed in enumerate(seeds):

        upRates, downRates = fetchRecentSR(results[i], showEpisodes=None, slideLen=slideLen)
        n_episodes = [i for i in range(len(upRates))]
        
        row,col = int(i/5),int(i%5)
        ax = axs[row][col]
        ax.set_title("SEED:{}".format(seed))
        ax.plot(n_episodes, upRates, label = "upCount/upTarget")
        ax.plot(n_episodes, downRates, label = "downCount/downTarget")
        if i==0 or i==5:
            ax.set_ylabel('success rate')

    plt.legend()
    plt.suptitle("{} Results with slideLen={}".format(envName, slideLen))
    plt.savefig("{}{}_trainResults.png".format(save_dir,envName),dpi=400)
        # plt.show()
    print("Save {}{}_trainResults.png successfully!".format(save_dir,envName))



def plot_uncertainty_results(showResult, method, policy, timeSteps, seeds, save_dir,max_num=5000):
    plt.figure()
    print('v1 -> Both, v2 -> Down, v3 -> Up')
    episode_num = max_num
    showResultDic = {
        'result':['MyRandomGridWorld-v1','MyRandomGridWorld-v2','MyRandomGridWorld-v3'],
        'downRate':['MyRandomGridWorld-v1','MyRandomGridWorld-v2'],
        'upRate':['MyRandomGridWorld-v1','MyRandomGridWorld-v3']
    }
    for envName in showResultDic[showResult]:
        xlimit, profile_dic, all_results = fetchProfileDF(method, policy, envName, timeSteps, seeds, save_dir, isRandom=False)
        episode_num = min(xlimit, episode_num)
        if showResult=='result':
            old_df = createPlotData(seeds, episode_num, all_results, showResult)
            new_df = create_new_df(old_df)
            sns.lineplot(data = new_df, x='idx', y='result', estimator='mean',label=envName.split('-')[-1])
        elif showResult=='downRate':
            assert envName[-1] in ['1','2'],"envNames should be in [v1, v2] to plot downRate result"
            old_df = createPlotData(seeds, episode_num, all_results, showResult)
            new_df = create_new_df(old_df)
            sns.lineplot(data = new_df, x='idx', y='result', estimator='mean',label=envName.split('-')[-1])
        elif showResult=='upRate':
            assert envName[-1] in ['1','3'],"envNames should be in [v1, v3] to plot upRate result"
            old_df = createPlotData(seeds, episode_num, all_results, showResult)
            new_df = create_new_df(old_df)
            sns.lineplot(data = new_df, x='idx', y='result', estimator='mean',label=envName.split('-')[-1])
        else:
            assert False,"undefined showResult:{}".format(showResult)

    plt.title("{}_{}".format(method, showResult))
    plt.legend(loc=4)# pos: lower right
    plt.savefig("{}{}_comparison.png".format(save_dir, showResult), dpi=300)
    print("Save {}{}_comparison.png successfully!".format(save_dir, showResult))
    # plt.show()