# coding:utf-8
import os
import pandas as pd
import shutil
import utils.file_utils as file_utils
'''
    根据Excel文件建立训练集
'''

source_excel_path = os.path.join(r"/home/amhs/toolid/Excel/20240904.xlsx")
train_base_dir = os.path.join(r"/home/amhs/toolid/bak/train")  # 训练文件目录
valid_base_dir = os.path.join(r"/home/amhs/toolid/bak/valid")

def gen_txt(excel_path, train_dir):
    dict_cnt = {} 
    dict_dict = {}
    df = pd.read_excel(excel_path)

    count = 0
    # 逐行迭代处理
    for index, row in df.iterrows():
        # 分类代码和分类名称
        code = row[13]
        name = row[81]
        val = 0
        img_path = row[6].replace('/image01/ADMSPA01', '/home/amhs/toolid/train')

        if len(dict_cnt) > 0:
            val = dict_cnt.get(code, 0)

        # 超过一千个图片就不处理了
        if val is not None and val > 99:
            continue

        #替换 linux环境处理 还是用code
        path = os.path.join(train_dir, str(code))
        if not file_utils.is_dir_exists(path):
            os.makedirs(path)

        # 找不到文件的情况下，就不处理了
        if not file_utils.is_file_exists(img_path):
            print(img_path)
            # continue
        else:
            #shutil.move(img_path, path)
            shutil.copy(img_path, path)
            dict_dict[str(code)] = index

            # 写入字典val
            if val is None or val == 0:
                dict_cnt[code] = 1
            else:
                val = val + 1
                dict_cnt[code] = val

    print(dict_dict)        
        

if __name__ == '__main__':
    gen_txt(source_excel_path, train_base_dir)
