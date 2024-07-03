import json
import os
import re
from tqdm import tqdm

def parse_cdl(input_string):
    # 使用正则表达式查找各个部分
    patterns = {
        'construction_cdl': r'construction_cdl:\n(.*?)(?=\n\w+_cdl|\Z)',
        'image_cdl': r'image_cdl:\n(.*?)(?=\n\w+_cdl|\Z)',
        'text_cdl': r'text_cdl:\n(.*?)(?=\n\w+_cdl|\Z)',
        'goal_cdl': r'goal_cdl:\n(.*?)(?=\n\w+_cdl|\Z)'
    }
    
    results = {}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            # 将匹配的结果去除前后空格并存入字典
            results[key] = match.group(1).strip()
    
    return results
if __name__ == '__main__':
    predict_cdl = []
    with open('/research/zhangzr/Bunny/outputs/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_structure_only-sft2/formalgeo_merged.jsonl', 'r') as f:
        for line in f:
            predict_cdl.append(json.loads(line))
    
    for item in tqdm(predict_cdl):
        prob_id = item['question_id']
        pred_str = item['text']
        pred_dict = parse_cdl(pred_str)
        # print(pred_dict)
        # 加载 formalgeo7k数据
        root_path = '/research/zhangzr/Bunny/data/formalgeo7k/formalgeo7k_v2/problems'
        with open(os.path.join(root_path, f"{prob_id}.json"), 'r') as f:
            formal_data = json.load(f)
        formal_data['model_predict_consCDL'] = ''
        formal_data['model_predict_imageCDL'] = ''
        if 'construction_cdl' in pred_dict.keys():
            formal_data['model_predict_consCDL'] = pred_dict['construction_cdl']
        if 'image_cdl' in pred_dict.keys():
            formal_data['model_predict_imageCDL'] = pred_dict['image_cdl']
        with open(os.path.join(f'{root_path}_withPredCDL', f"{prob_id}.json"), 'w') as f:
            f.write(json.dumps(formal_data))
        
    
        