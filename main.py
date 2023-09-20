import jieba

import re
import numpy as np
from collections import defaultdict, OrderedDict


def get_file_contents(path):
    string = ''
    f = open(path, 'r', encoding='UTF-8')
    line = f.readline()
    while line:
        string = string + line
        line = f.readline()
    f.close()
    return string


def build_frequency_dict(tokens):
    frequency_dict = defaultdict(int)
    for token in tokens:
        frequency_dict[token] += 1
    return frequency_dict


def align_dictionaries(dict1, dict2):
    sorted_dict1 = OrderedDict(sorted(dict1.items()))
    sorted_dict2 = OrderedDict(sorted(dict2.items()))
    aligned_dict1 = OrderedDict()
    aligned_dict2 = OrderedDict()
    all_keys = set(sorted_dict1.keys()) | set(sorted_dict2.keys())
    for key in all_keys:
        aligned_dict1[key] = sorted_dict1.get(key, 0)
        aligned_dict2[key] = sorted_dict2.get(key, 0)
    return aligned_dict1, aligned_dict2


def filter(string):
    pattern = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
    string = pattern.sub("", string)
    result = jieba.lcut(string)
    return result


def calculate_simularity(text1, text2):
    dict1 = build_frequency_dict(text1)
    dict2 = build_frequency_dict(text2)

    # 对齐字典并按键排序
    aligned_dict1, aligned_dict2 = align_dictionaries(dict1, dict2)


    # 将字典的值转换为 NumPy 向量
    vector1 = np.array(list(aligned_dict1.values()))
    vector2 = np.array(list(aligned_dict2.values()))


    dot_product = np.dot(vector1, vector2)
    # 计算向量范数
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm1 * norm2)

    # 输出结果

    return cosine_similarity

def save_float_to_file(float_number, file_path):
    string = str(float_number)  # 将浮点数转换为字符串
    with open(file_path, 'w') as file:
        file.write(string)

def main():

    path1 = input("原文文件的绝对路径：")
    path2 = input("抄袭文件的绝对路径：")
    file_path = input("答案文件保存的绝对路径")
    str1 = get_file_contents(path1)
    str2 = get_file_contents(path2)
    text1 = filter(str1)
    text2 = filter(str2)
    similarity = calculate_simularity(text1, text2)
    print(similarity)
    save_float_to_file(similarity, file_path)
    return similarity


if __name__ == '__main__':
    main()
