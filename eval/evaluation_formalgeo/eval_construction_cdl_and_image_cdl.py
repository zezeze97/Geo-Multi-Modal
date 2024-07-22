import json
from utils import getConsCdlAcc, getImgCdlAcc
import re
import os
import numpy as np
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction

def percentage_of_perfect(lst):
    # 计算等于1.0的元素的数量
    count_of_perfect = len([item for item in lst if item >=1.0])
    # 获取列表的总长度
    total_elements = len(lst)
    # 计算占比，避免除以零的错误
    if total_elements > 0:
        return (count_of_perfect / total_elements) * 100
    else:
        return 0


def parse_cdl(input_string):
    # 使用正则表达式查找各个部分
    patterns = {
        'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'text_cdl': r'(?:The )?text_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)',
        'goal_cdl': r'(?:The )?goal_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)'
    }
    
    results = {}
    
    # 优先匹配包含"calibrate"的版本
    for key, pattern in patterns.items():
        pattern = pattern.replace("(?:calibrate )?", "(?:calibrate )")
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            results[key] = match.group(1).strip()
        else:
            # 如果未找到包含"calibrate"的版本，尝试匹配不含"calibrate"的版本
            pattern = pattern.replace("(?:calibrate )", "(?:calibrate )?")
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
    
    return results

def getScore(predict_file, gt_path):
    running_consCdl = []
    running_imageCdl = []
    running_textCdl = []
    running_goalCdl = []
    predict_lst = []
    with open(predict_file, 'r') as f:
        for line in f:
            predict = json.loads(line)
            predict_lst.append(predict)
    for predict in predict_lst:
        qs_id = predict['question_id']
        response = predict['text']
        result = parse_cdl(response)
        print(result)
        if 'construction_cdl' not in result.keys():
            p_construction_cdl = ''
        else:
            p_construction_cdl = result['construction_cdl'].replace(', ', ',')
            
            
        if 'image_cdl' not in result.keys():
            p_image_cdl = ''
        else:
            p_image_cdl = result['image_cdl'].replace(', ', ',')
            
            
        if 'text_cdl' not in result.keys():
            p_text_cdl = ''
        else:
            p_text_cdl = result['text_cdl'].replace(', ', ',')
        
        if 'goal_cdl' not in result.keys():
            p_goal_cdl = ''
        else:
            p_goal_cdl = result['goal_cdl']
        
        
        with open(os.path.join(gt_path, f'{qs_id}.json'), 'r') as f:
            gt_info = json.load(f)
        gt_construction_cdl = ','.join(gt_info['construction_cdl'])
        gt_image_cdl = ','.join(gt_info['image_cdl'])
        gt_text_cdl = ','.join(gt_info['text_cdl'])
        gt_goal_cdl = gt_info['goal_cdl']
        
        # print(f'p_goal_cdl: {p_goal_cdl}, gt_goal_cdl:{gt_goal_cdl}')
        
        if p_text_cdl == gt_text_cdl:
            running_textCdl.append(1.0)
        else:
            running_textCdl.append(0.0)
        
        if p_goal_cdl == gt_goal_cdl:
            running_goalCdl.append(1.0)
        else:
            running_goalCdl.append(0.0)
        
    
        try:
            consCdlAcc, _ = getConsCdlAcc(gt_construction_cdl, p_construction_cdl)
            # print(f'consCdlAcc is {consCdlAcc}')
            
        except:
            consCdlAcc = 0.0       
            
        try:
            imageCdlAcc, _ = getImgCdlAcc(qs_id, gt_image_cdl,  p_image_cdl)
            # print(f'imageCdlAcc is {imageCdlAcc}')
        except:
            imageCdlAcc = 0.0
        
        running_consCdl.append(consCdlAcc)
        running_imageCdl.append(imageCdlAcc)
    
    
    
    print(f'Average construction_cdl acc is {np.mean(running_consCdl) * 100}\nPerfect construction_cdl is {percentage_of_perfect(running_consCdl)}')
    print(f'Average image_cdl acc is {np.mean(running_imageCdl) * 100}\nPerfect image_cdl is {percentage_of_perfect(running_imageCdl)}')
    print(f'textCdlPerfect is {np.mean(running_textCdl) * 100}\ngoalCdlPerfect is {np.mean(running_goalCdl) * 100}')
            
    assert len(running_consCdl) == len(running_imageCdl)
    num_of_both_perfect = 0.0
    for i in range(len(running_consCdl)):
        num = running_consCdl[i] + running_imageCdl[i]
        if num >=2.0:
            num_of_both_perfect += 1.0
    both_perfect = (num_of_both_perfect / len(running_consCdl)) * 100
    print(f'Both perfect construction_cdl and image_cdl: {both_perfect}')
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str)
    parser.add_argument('--gt-path', default='data/formalgeo7k/formalgeo7k_v2/problems')
    args = parser.parse_args()
    
    getScore(args.prediction, args.gt_path)
    
    