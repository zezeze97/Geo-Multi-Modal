import argparse
import json
import re
import os

def extract_choice(q, answer, i):
    
    
    
    if not args.choice_mode:
        # 使用正则表达式查找"The final answer is:"后的数字
        match = re.search(r"The final answer is:\s*(.*)", answer)
        if match:
            return match.group(1)
    else:
        
        pattern0 = r"The final answer is: ([A-D])"
        match = re.search(pattern0, answer)
        if match:
            return match.group(1)
    
        pattern0 = r"The answer is option ([A-D])"
        match = re.search(pattern0, answer)
        if match:
            return match.group(1)
        
        
        
        
        pattern0 = r"Answer:([A-D])"
        match = re.search(pattern0, answer)
        if match:
            return match.group(1)
        # Pattern 1: Therefore, the correct answer is option {choice}

        pattern1 = r"the correct answer is option ([A-D])"
        match = re.search(pattern1, answer)
        if match:
            return match.group(1)

        # Pattern 2: Therefore, option {choice} is selected
        pattern2 = r"option ([A-D]) is selected"
        match = re.search(pattern2, answer)
        if match:
            return match.group(1)

        # Pattern 3: Therefore, the answer is option {choice}
        pattern3 = r"the answer is option ([A-D])"
        match = re.search(pattern3, answer)
        if match:
            return match.group(1)
        #
        # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
        pattern4 = r"the answer is \(([A-D])\)"
        match = re.search(pattern4, answer)
        if match:
            return match.group(1)

        # Pattern 5: find the solution from the last sentence
        sentences = answer.split(".")
        try:
            last_sentence = sentences[-2].strip()
        except:
            return None
        match = re.search(r'is ([A-D])', last_sentence)
        if match:
            answer = match.group(1)
            return answer
    # print(f"not found {i}")
    # print("#"*10)
    # print(f"question:\n{q}")
    # print("#"*10)
    # print(f"answer:\n{answer}")
    # print("#"*10)
    return None

def paring_origin_qa(origin_qa):
    origin_qs = origin_qa['Question'] # '如图所示,在▱ABCD中,已知AD=10cm,AB=4cm,AE平分∠BAD交BC于点E,则EC等于()'
    origin_ans = origin_qa['Solution'] # '解:在▱ABCD中,AD=BC,AD∥BC,∴∠DAE=∠BEA,∵AE平分∠BAD交BC于点E,∴∠BAE=∠DAE,∴∠BAE=∠BEA,∴AB=BE,∵AD=10cm,AB=4cm,∴AB=10cm,BE=4cm,∴EC=6cm．故选:B．'
    origin_comment = None
    if 'Comment' in origin_qa.keys():
        origin_comment =origin_qa['Comment'] # '利用平行线和角平分线得到等角,进而得到等腰三角形,再利用等腰三角形的性质解题,是几何中的常见题目．'
    
    choices = origin_qa['Choices']
    select_choices = origin_qa['Label']
    map_choices = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    select_choices =map_choices[str(select_choices)]
    return origin_qs, origin_ans, origin_comment, choices, select_choices


def calculate_accuracy(args):
    with open(args.predictions_file, "r") as f_pred:
        predictions = [json.loads(line) for line in f_pred]
        predictions_ids = [int(item["question_id"]) for item in predictions]
        questions = [item["prompt"] for item in predictions]
        predictions = [item["text"] for item in predictions]
        _ground_truth_info = []
        for prob_id in predictions_ids:
            if args.choice_mode:
                with open(os.path.join(f'{args.ground_truth_path}_withAns', f'{prob_id}.json'), 'r') as f:
                    _ground_truth_info.append(json.load(f))
                
            else:
                with open(os.path.join(args.ground_truth_path, f'{prob_id}.json'), 'r') as f:
                    _ground_truth_info.append(json.load(f))
        ground_truth_dict = {}
        for grd in _ground_truth_info:
            if args.choice_mode:
                origin_qa = grd["origin_qa"]
                _, _, _, _, ans = paring_origin_qa(origin_qa)
                ground_truth_dict.update({int(grd["problem_id"]) : ans})
            else:
                ground_truth_dict.update({int(grd["problem_id"]) : grd["problem_answer"]})
        ground_truth = [ground_truth_dict[i] for i in predictions_ids]
        origin_source_dict = {}
        for grd in _ground_truth_info:
            origin_source_dict.update({int(grd["problem_id"]) : grd["source"]})
        sources = [origin_source_dict[i] for i in predictions_ids]
        # Extract choices from predictions
        predicted_ans = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions))]
        none_num = len([x for x in predicted_ans if x is None])
        print(f"nones : {none_num}")

        # Calculate accuracy
        total = len(ground_truth)
        correct = 0
        correct_list, wrong_list = [], []
        for prob_id, source, q, gt, pred_a, pred in zip(predictions_ids, sources, questions, ground_truth, predicted_ans, predictions):
            if pred_a == gt:
                correct+=1
                correct_list.append({"probID": prob_id, "source": source, "question": q, "pred": pred, "gt": gt})
            else:
                wrong_list.append({"probID": prob_id, "source": source, "question": q, "pred": pred, "gt": gt})
        accuracy = correct / total * 100
        if args.save_correct_wrong:
            correct_file, wrong_file = os.path.splitext(args.predictions_file)[0]+"_correct.json", os.path.splitext(args.predictions_file)[0]+"_wrong.json"
            with open(correct_file, "w") as f:
                json.dump(correct_list, f)
            with open(wrong_file, "w") as f:
                json.dump(wrong_list, f)

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_path", type=str, default="data_backup/formalgeo7k/formalgeo7k_v2/problems")
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--save_correct_wrong", action="store_true")
    parser.add_argument("--choice_mode", action='store_true')
    args = parser.parse_args()
    accuracy = calculate_accuracy(args)
    print(f"Accuracy: {accuracy:.2f}%")