
import time
import joblib
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from override.ppo2 import PPO2
from override.pposgd_simple import PPO1
from override.dqn import DQN


# %matplotlib inline
from utils import analysis_trajectory
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 评估现有model在环境env下的性能表现，计算n_episode获得的平均reward。
def playGridWorldGame(env, model, n_episodes=100, show= False):
    # print(">>>>>>>Play Games in {} with {}(random:{})...".format(envName,method,isRandom))   envName, method, isRandom ,
    plt.axis("off")
    plt.figure(figsize=(10,10))
    upTarget = 0
    downTarget = 0
    fail = 0
    all_episode_rewards = []
    episode_rewards = None
    upcount = 0
    for i in range(n_episodes):
        obs = env.reset()
        if env.ends[0][-1]==(env.n_height-1):
            upcount += 1
        done = False
        count = 0

        episode_rewards = []
        if show:
            img = plt.imshow(env.render(mode='rgb_array')) 
            img.set_data(env.render(mode='rgb_array'))
            display.display(plt.gcf())
            display.clear_output(wait=True)
        while True:
            count += 1
            action = model.predict(obs)[0]
            obs,reward,done,info = env.step(action)
            episode_rewards.append(reward)
            if show:
                img.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)

            if done and reward<0:
                fail += 1
                break

            if done and reward>0:
                if obs[0]==0 or obs[1]==0:
                    downTarget += 1
                elif obs[0] == obs[-2] or obs[1] == obs[-1]:
                    upTarget += 1
                break
        all_episode_rewards.append(sum(episode_rewards))


    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print("Finish Game {} Times".format(n_episodes))
    print("upTargets:{}/{}".format(upTarget,upcount))
    print("downTargets:",downTarget)
    print("fail:",fail)
    print("Mean reward:", mean_episode_reward)
    print("Std reward:", std_episode_reward)

    return all_episode_rewards, episode_rewards


# 训练或加载模型，返回模型训练过程中的统计信息。
def train_or_load_agent(seed, policy, envName, method, save_dir, timeSteps, load = True, isRandom = False, winReward=1, buffer_size = None ):
    
    import gym,os
    import myenv

    print(">>>>>>>>>>",os.getpid(),policy, envName, method, timeSteps,seed)
    cur_time = time.time()
    env = gym.make(envName,seed=seed)
    os.makedirs(save_dir, exist_ok=True)
    saveanme = "{}_{}_{}_{}_{}_{}".format(method, policy, envName, int(isRandom), timeSteps, seed)
    savepath = os.path.join(save_dir,saveanme)
    if os.path.exists(savepath+".zip") and load:
        print("$$$\tLoad pretrained model...")
        model = eval(method).load(savepath)
        print("$$$\tload successfully:",savepath)
    else:
        if isRandom:
            model = eval(method)(policy, env, verbose=0) # TODO
        else:
            if method == "DQN":
                assert buffer_size != -1,"please assign buffer_size for DQN"
                model, trajectory_dic  = eval(method)(policy, env, buffer_size=buffer_size, verbose=0, seed=seed, save_trajectory = True).learn(timeSteps)
            else:
                model, trajectory_dic  = eval(method)(policy, env, verbose=0, seed=seed, save_trajectory = True).learn(timeSteps)
        model.save(savepath)
        joblib.dump(trajectory_dic,savepath+"_traj.dic")
        print("$$$\ttrain and save a {} model(random:{}) successfully:".format(method, isRandom))
        print("$$$\ttrajectories are saved in: ",savepath)
    print(">>>>>>>>>>",np.around(time.time()-cur_time,2))
    env.close()
    results_dic = analysis_trajectory(trajectory_dic, savepath, winReward)
    return results_dic

