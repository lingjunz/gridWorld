### 1. Description
This is a demo DRL respository to discover interesting phenomena with typical DRL algorithms (i.e., PPO1,, PPO2, DQN) in customized gridWorld environment.

The DRL frameworsk used is [Stable Baselines(v2.10.0)](https://stable-baselines.readthedocs.io/en/master/).


### 2. Deployments

* `sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`

#### 2.1 安装 Anaconda
* `wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
* `bash ./Anaconda3-2020.11-Linux-x86_64.sh`
* (optional) `conda config --set auto_activate_base false`
* (should be changed)`echo 'export PATH="$pathToAnaconda/anaconda3/bin:$PATH"' >> ~/.bashrc`

#### 2.2 配置jupyter notebook
* 生成配置文件：`jupyter notebook --generate-config`
* 设置密码：`jupyter notebook password`
* 修改配置文件：`vim ~/.jupyter/jupyter_notebook_config.py`
```
# 添加(覆盖)下面内容
c.NotebookApp.ip = '*' # 开启所有的IP访问，即可使用远程访问
c.NotebookApp.open_browser = False # 关闭启动后的自动开启浏览器
c.NotebookApp.port = 8888  # 设置端口8888，也可用其他的，比如1080，8080等等
```
* 创建conda虚拟环境
    * 导出已有环境：在虚拟环境中执行`conda env export > grid.yaml`
    * 根据导出文件创建环境：`conda env create -f grid.yaml`
* 配置jupyter notebook kernel
    * `conda install ipykernel`
    * 激活虚拟环境，将环境写入notebook的kernel中,`python -m ipykernel install --user --name grid --display-name grid`


### Directories AND Files

```
├── Readme.md                      // help
├── myenv                          
|   ├── grid_world.py              // customized environments(MyRandomGridWorld-1,2,3)
|   ├── __init__.py                // registries
├── saved                          // the saved files for analysis
|
│── command.sh                     // examples of command
|
|── main.py                        // entry point to run experiments
|
|── playGridWorld.py               // functions to collect a model and play games
|
|── plot_func.py                   // functions to plot results and save figures
|
|── utils.py                       // utility functions for analysis
|
|── tryGridWorld.ipynb             // jupyter notebook for convenient interactions  

```

