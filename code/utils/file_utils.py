# -*- coding:utf-8 -*-
"""
@file name  : file_utils.py
@author     : Chris.Ma
@date       : 2024-05-30
@brief      : 读取文件和解析文件要素
"""
import shutil
import os
from typing import List


class DefectList:
    # 一张图片可能有多个defectlist
    def __init__(self, file_name):
        self.file_name = file_name
        self.defect_list = []
    
    def add_defect(self, element):
        self.defect_list.append(element)

def get_files_by_suffix(directory, suffixes):
    matching_files = []
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1].lstrip('.') in suffixes:
            matching_files.append(os.path.join(directory, filename))
    return matching_files

def get_layer_from_file(file_path):
    """
    从文件中获取layer
    :param file_path:  文件全路径
    :return: 从文件中获取的layer
    """    
    last_columns = "" 
    with open(file_path, "r") as file:
        for line in file:
            print(line)
            # 判断字符串是否以特定的子字符串开头
            tmp = line.replace('\r', '').replace('\n', '')
            if tmp.startswith("StepID"):
                last_columns = tmp.rstrip().rsplit(maxsplit=1)[-1].replace('"', '').replace(';', '')
                break
    return last_columns


def get_defect_from_file(file_path) -> List[DefectList]:
    array_list = []
    is_begin = False
    global tmp_stru    

    with open(file_path, "r") as file:
        for line in file:
            tmp = line.replace('\r', '').replace('\n', '')

            # 最后一条数据了
            if (is_begin == True and tmp.startswith("SummarySpec")):
                array_list.append(tmp_stru)
                is_begin = False

            # 判断字符串是否以特定的子字符串开头
            if tmp.startswith("TiffFileName"):
                # 上一条defectlist数据获取结束，添加到数组中
                if (is_begin == True):
                    array_list.append(tmp_stru)
                    is_begin = False

                file_name = tmp.rstrip().rsplit(maxsplit=1)[-1].replace(';', '')
                tmp_stru = DefectList(file_name)
                
            # 要开始读取下一行的defectList了
            if tmp.startswith("DefectList") and not tmp.startswith("DefectList;"):
                is_begin = True
                continue

            if (is_begin == True):
                tmp_stru.add_defect(tmp)

    return array_list

def get_dict_from_file(file_path):
    dict = {}
    is_begin = False

    with open(file_path, "r") as file:
         for line in file:
            tmp = line.replace('\r', '').replace('\n', '')
            if tmp.startswith("ClassLookup"):
                is_begin = True
                continue
            
            if (is_begin == True and tmp.startswith(" ")):
                tmp_dict = tmp.split()
                dict[int(tmp_dict[0])] = tmp_dict[1].replace('"', '')

            # 结束循环
            if (is_begin == True and not tmp.startswith(" ")):
                is_begin = False
                break
    
    return dict

def get_classnumber_from_line(defect_line, column_number):
    tmp_columns = defect_line.split()
    if (bool(tmp_columns) is False):
        return -1 
    
    if (len(tmp_columns) < column_number):
        return -1
    
    return tmp_columns[column_number - 1]

def is_dir_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def is_file_exists(file):
    return os.path.exists(file) and os.path.isfile(file)

def move_file(root, file, subdir):
    fulldir = os.path.join(root, subdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir,  exist_ok=False)
    
    fullfile = os.path.join(root, file)
    shutil.move(fullfile, fulldir)

if __name__ == "__main__":
    # print(get_layer_from_file(r"D:\Test\walfa\AYSEV01\A005169#010827084553.001"))
    for defect in get_defect_from_file(r"D:\Test\walfa\AYSEV01\A005169#010827084553.001"):
        print(defect.file_name)
        for temp in defect.defect_list:
            print(temp)

    # print(get_dict_from_file(r"D:\Test\walfa\AYSEV01\A005169#010827084553.001"))
    # print(get_classnumber_from_line(" 41 3466.882 1522.202 -7 34 3.750 3.200 12.000000 3.750 13 1 13 0 0.000000 0 0.000000 0.000000 0.000000 0.000000 0.000000 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 4 4 1 0 2 0 3 0 4 0;", 10));

    # suffix_list = ['001', '000'] 
    # print(get_files_by_suffix(r"D:\Test\walfa\AYSEV01", suffix_list))
