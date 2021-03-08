

import time
import argparse
import multiprocessing
from functools import  partial
from plot_func import plot_all_results
from playGridWorld import train_or_load_agent
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# xvfb-run -s "-screen 0 1400x900x24" jupyter notebook

# python main.py 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help = "RL alg. to train an agent", choices = ['PPO1','PPO2','DQN'], default = 'PPO1')
    parser.add_argument("--envName", help = "MyRandomGridWorld-v1,2,3, v1 -> Both, v2 -> Down, v3 -> Up", 
                                    choices = ['MyRandomGridWorld-v1','MyRandomGridWorld-v2','MyRandomGridWorld-v3'],
                                    default = 'MyRandomGridWorld-v2' )
    parser.add_argument("--save_dir", help = "path to save models and analysis results", type = str, default = "./saved/")
    parser.add_argument("--isRandom", help = "not use", type = bool, default = False)
    parser.add_argument("--policy", help = "not use", choices = ['MlpPolicy'], default = 'MlpPolicy')
    parser.add_argument("--timeSteps", help = "timeSteps for training", type = int, default = 200000)
    parser.add_argument("--start_seed", help = "start seed", type = int, default = 111)
    parser.add_argument("--end_seed", help = "end seed", type = int, default= 121)
    parser.add_argument("--buffer_size", help = "buffer_size for DQN", type = int, default = -1)
    args = parser.parse_args()

    method = args.method
    envName = args.envName
    timeSteps = args.timeSteps 
    if args.buffer_size != -1:
        save_dir = "{}{}/{}/".format(args.save_dir, method, args.buffer_size) 
    else:
        save_dir = args.save_dir + method +"/"
    buffer_size = args.buffer_size
    seeds = [i for i in range(args.start_seed,args.end_seed)]
    start_time = time.time()


    p = multiprocessing.Pool(5)
    results = p.map(partial(train_or_load_agent, policy = args.policy, envName = envName,
                                       method = method, save_dir = save_dir, timeSteps = timeSteps,
                                       load = False, isRandom = False, winReward=1, buffer_size = buffer_size), seeds)
    p.close()
    p.join()

    print(">>>Time:{:.2f}".format(time.time()-start_time))
    print("\n\n")
    

    plot_all_results(envName, save_dir,seeds,results,slideLen = 100)

    


    
