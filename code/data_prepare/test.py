import os
 
def loop_files_in_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=True):
        for file in files:
            print(os.path.join(root, file))
            print(file)

def mkdir(root, subdir):
    fulldir = os.path.join(root, subdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir,  exist_ok=False)
    
def dir_only_files(directory):
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            print(os.path.join(directory, file))

# 使用示例
dir_only_files('D:\\pyspaces\\recz\\data\\test')
 