import re
import json
import os
from sklearn.metrics import f1_score,accuracy_score

datasets=['scifact','fever','vc']
shots=[3,6,12,24,48]
K=[1,2,4,8,16]
seeds=[0,2,3,4]

pred_mapping={0:'SUPPORT',1:'NEI',2:'REFUTE'}
for dataset in datasets:    
    for sh in shots:
        for seed in seeds:          
            predictions=[]
            labels=[]
            idxs=[]
            #output/vc_shots48_seed3_fs_stage2/test_pred.txt
            filename="output/{}_shots{}_seed{}_fs_stage2/test_pred.txt".format(dataset,sh,seed)
            print(f"dealing with{filename}. ",filename)
            # wrong_idx=[]
            # wrong_pred=[]
            wrongs={}
            logged_f1=0
            try:
                with open(filename,'r') as f:
                    for line in f.readlines():
                        if 'F1' in line:
                            numbers = re.findall(r'\d+\.\d+', line)
                            logged_f1=numbers[0]
                        if 'prediction:' in line:
                            # 使用正则表达式从行中提取数字
                            numbers = re.findall(r'\d+', line)
                            # 将数字转换为整数并添加到列表中
                            predictions.extend([int(num) for num in numbers])
                        if 'labels' in line:
                            # 使用正则表达式从行中提取数字
                            numbers = re.findall(r'\d+', line)
                            # 将数字转换为整数并添加到列表中
                            labels.extend([int(num) for num in numbers])
                        if 'idx' in line:
                            # 使用正则表达式从行中提取数字
                            numbers = re.findall(r'\d+', line)
                            # 将数字转换为整数并添加到列表中
                            idxs.extend([int(num) for num in numbers])
                assert len(predictions)==len(labels)
                print(f"logged f1:{logged_f1}",end=', calculated f1: ')
                print(f1_score(labels,predictions,average='macro'))
                print(f"cal acc:{accuracy_score(labels,predictions)}")
                for i in range(len(predictions)):
                    if predictions[i]!=labels[i]:
                        # wrong_pred.append(predictions[i])
                        # wrong_idx.append(idxs[i])
                        wrongs[idxs[i]]=predictions[i]
            except FileNotFoundError:
                print(f" not found  {filename}.",filename)                
                continue
            try:
                validation_file = 'data_dir/{}_validation.jsonl'.format(dataset)
                output_file = 'bad_case/{}_shots{}_seed{}_fs_stage2.jsonl'.format(dataset, sh, seed)
                print(f"wrong case number:{len(wrongs.keys())}")
                # 打开验证文件
                with open(validation_file, 'r') as f_val:                    
                    with open(output_file, 'w') as f_out:
                        for line in f_val.readlines():
                            item = json.loads(line.strip())                                
                            if item['id'] in wrongs.keys():
                                item['wrong_pred'] = pred_mapping[wrongs[item['id']]]
                                f_out.write(json.dumps(item) + '\n')
                
                    
                        
            except FileNotFoundError as e:
                print(e.filename)
                # print(" not found  data_dir/{}_validation.jsonl'.format(dataset)")                
                continue