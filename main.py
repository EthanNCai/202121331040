import jieba
import re
import numpy as np
import sys
from collections import defaultdict, OrderedDict

#这个函数是用来读取文件的
def get_file_contents(path):
    try:
        with open(path, 'r', encoding='UTF-8') as f:
            return f.read()
    except IOError:
        print(f"Error:文件读取失败，位置： {path}")
        return ""

#使用字典来初步构建分词的词频表
def build_frequency_dict(tokens):
    frequency_dict = defaultdict(int)
    for token in tokens:
        frequency_dict[token] += 1
    return frequency_dict

#使用有序字典，将两个字典处理为维度相同的向量
def align_dictionaries(dict1, dict2):
    sorted_dict1 = OrderedDict(sorted(dict1.items()))
    sorted_dict2 = OrderedDict(sorted(dict2.items()))
    aligned_dict1 = OrderedDict()
    aligned_dict2 = OrderedDict()
    all_keys = set(sorted_dict1.keys()) | set(sorted_dict2.keys())
    for key in all_keys:
        //如果在文件b的某一个词在文件a里未出现，那么a的向量的该位置为0
        aligned_dict1[key] = sorted_dict1.get(key, 0)
        aligned_dict2[key] = sorted_dict2.get(key, 0)
    return aligned_dict1, aligned_dict2

# 用正则表达式初步规律以及使用jieba库进行自动分词
def filter_text(string):
    pattern = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
    string = pattern.sub("", string)
    result = jieba.lcut(string)
    return result

# 基于余弦函数的空间相似度计算
def calculate_similarity(text1, text2):
    dict1 = build_frequency_dict(text1)
    dict2 = build_frequency_dict(text2)
    aligned_dict1, aligned_dict2 = align_dictionaries(dict1, dict2)
    vector1 = np.array(list(aligned_dict1.values()))
    vector2 = np.array(list(aligned_dict2.values()))
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

## 保存文件
def save_float_to_file(float_number, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(str(float_number))
    except IOError:
        print(f"Error: 文件写入失败，位置： {file_path}")


def main():
    path1 = sys.argv[1]  # 原文文件路径
    path2 = sys.argv[2]  # 抄袭版论文文件路径
    file_path = sys.argv[3]  # 答案文件路径

    str1 = get_file_contents(path1)
    str2 = get_file_contents(path2)

    if not str1 or not str2:
        print("Error: 空文件，程序即将终止")
        return

    text1 = filter_text(str1)
    text2 = filter_text(str2)

    similarity = calculate_similarity(text1, text2)
    print("相似度:", similarity)
    save_float_to_file(similarity, file_path)


if __name__ == '__main__':
    main()
