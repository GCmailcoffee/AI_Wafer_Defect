# coding:utf-8
import os
import pandas as pd
import shutil
import utils.file_utils as file_utils
'''
    根据Excel文件建立训练集
'''

source_excel_path = os.path.join(r"/home/amhs/toolid/Excel/20240904.xlsx")
train_base_dir = os.path.join(r"/home/amhs/toolid/training")  # 训练文件目录
valid_base_dir = os.path.join(r"/home/amhs/toolid/valid")


def gen_txt(excel_path, train_dir):
    dict_cnt = {} 
    df = pd.read_excel(excel_path)

    # 逐行迭代处理
    for index, row in df.iterrows():
        # 分类代码和分类名称
        code = row[13]
        name = row[81]
        val = 0
        img_path = row[6].replace('/image01/ADMSPA01', '/home/amhs/toolid/train')

        # 找不到文件的情况下，就不处理了
        if not file_utils.is_file_exists(img_path):
            continue
        else:
            #shutil.move(img_path, path)
            val =  dict_cnt.get(name, 0)
            # 写入字典val
            if val is None or val == 0:
                dict_cnt[name] = 1
            else:
                val = val + 1
                dict_cnt[name] = val

    print(dict_cnt)        
        

if __name__ == '__main__':
    gen_txt(source_excel_path, train_base_dir)
