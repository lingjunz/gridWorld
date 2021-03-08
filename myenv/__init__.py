from gym.envs.registration import register


register(
    id = 'MyRandomGridWorld-v0', # 环境名,版本号v0必须有
    entry_point = 'myenv.grid_world:MyRandomGridWorld0', # 文件夹名.文件名:类名
    # 根据需要定义其他参数
    
)

register(
    id = 'MyRandomGridWorld-v1', # 环境名,版本号v0必须有
    entry_point = 'myenv.grid_world:MyRandomGridWorld1', # 文件夹名.文件名:类名
    # 根据需要定义其他参数
)

register(
    id = 'MyRandomGridWorld-v2', # 环境名,版本号v0必须有
    entry_point = 'myenv.grid_world:MyRandomGridWorld2', # 文件夹名.文件名:类名
    # 根据需要定义其他参数

)

register(
    id = 'MyRandomGridWorld-v3', # 环境名,版本号v0必须有
    entry_point = 'myenv.grid_world:MyRandomGridWorld3', # 文件夹名.文件名:类名
    # 根据需要定义其他参数

)
