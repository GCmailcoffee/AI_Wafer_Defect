# coding:utf-8
import os
import random
'''
    生成txt文件
'''


result_txt_path = os.path.join("D:\\pyspaces\\recz\\data\\result\\result.txt")
test_dir = os.path.join("D:\\back\\work")


def gen_result(txt_path, img_dir):
    f = open(txt_path, 'w')
    
    for root, s_dirs, files in os.walk(img_dir, topdown=True):  
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
            
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):         # 若不是png文件，跳过
                    continue
                
                img_path = os.path.join("D:\\谷歌下载\\跑批200张\\", img_list[i])
                line = img_path + ' ' + sub_dir + ' ' + str(random_float_between(35.01, 58.49)) + '\n'
                f.write(line)
    f.close()

def random_float_between(a, b):
    return round(random.uniform(a, b), 2)

if __name__ == '__main__':
    gen_result(result_txt_path, test_dir)
    