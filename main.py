from model import qwen, LLM, llama

from vlmeval.dataset import build_dataset
from mctstask import MCTS_Task
from node import treeNode
from mcts import selectNode, get_next_steps_expand, expand

import ast
import torch
import os
import os.path as osp
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from easydict import EasyDict
# from model import clip
# from model import llm



def run(args):
    os.environ["OPENAI_API_KEY"] = 
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1/chat/completions" # Replace with your actual base
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")    
    print('-'*30, 'Begin testing', '-'*30, '\n')
    try:
        dataset_kwargs = {}
        dataset_name = args.dataset#"MathVista_MINI"
        dataset = build_dataset(dataset_name, **dataset_kwargs)
        print(f'전체 데이터셋 길이:{len(dataset.data)}')
        dataset.data = dataset.data.iloc[:10]
        data_len = len(dataset.data)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    model, processor = llama('llama')
    # model, processor = qwen('Qwen2_5')
    model_dict = LLM('qwen')
    output_list = []
    correct_count = 0
    dataset.data['prediction'] = ''
    for i in range(len(dataset.data)):
        image = dataset.data.iloc[i]['image']
        dataset.dump_image(dataset.data.iloc[i])
        img = osp.join(dataset.img_root, f"{dataset.data.iloc[i]['index']}.jpg")
        question = dataset.data.iloc[i]['question']
        if not pd.isnull(dataset.data.iloc[i]['choices']):
            choices = dataset.data.iloc[i]['choices']

# base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local'], default='glm')
# base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
        Task = MCTS_Task(question, model, processor, args.propose_method, args.value_method, args.branch, args.end_gate,
                            args.roll_policy, args.roll_branch, args.roll_forward_steps, args.time_limit,
                            args.iteration_limit, args.exploration_constant, args.alpha, args.inf,
                            args.temperature, use_case_prompt=args.use_case_prompt, use_reflection=args.use_reflection,
                            low=args.low, high=args.high, evaluate=args.evaluate,img_path=img,model_dict=model_dict)
        output, root = Task.run()
        print('content:')
        print(output['content'])
        print('\nsolution:')
        print(output['solution'])
        print('\nsummary:')
        print(output['summary'])
        sorted_samples = sorted(output['value_samples'], key=lambda x: x['value'], reverse=True)
        second_highest_steps = sorted_samples[1]['steps']
        if len(sorted_samples) > 1:
            print('\nsecond_solution:')
            print(second_highest_steps)
        # evaluate metrics
        if args.evaluate:
            
            # labels = ["(A)", "(B)", "(C)", "(D)"]
            # mapped_choices = dict(zip(labels, dataset.data.iloc[i]['choices']))
            # print(mapped_choices)
            if len(output['summary']) < 5 and ('A' in output['summary'] or '(A' in output['summary']):
                real_list = ast.literal_eval(dataset.data.iloc[i]['choices'])
                # dataset.data.iloc[i]['prediction'] = real_list[0]
                dataset.data.loc[i, 'prediction'] = real_list[0]
            elif len(output['summary']) < 5 and ('B' in output['summary'] or '(B' in output['summary']):
                real_list = ast.literal_eval(dataset.data.iloc[i]['choices'])
                # dataset.data.iloc[i]['prediction'] = real_list[1]
                dataset.data.loc[i, 'prediction'] = real_list[1]
                
            elif len(output['summary']) < 5 and ('C' in output['summary'] or '(C' in output['summary']):
                real_list = ast.literal_eval(dataset.data.iloc[i]['choices'])
                # dataset.data.iloc[i]['prediction'] = real_list[2]
                dataset.data.loc[i, 'prediction'] = real_list[2]
            elif len(output['summary']) < 5 and ('D' in output['summary'] or '(D' in output['summary']):
                real_list = ast.literal_eval(dataset.data.iloc[i]['choices'])
                # dataset.data.iloc[i]['prediction'] = real_list[3]
                dataset.data.loc[i, 'prediction'] = real_list[3]
            else:
                dataset.data.loc[i, 'prediction'] = output['summary']
            pre = dataset.data.iloc[i]['prediction']
            print(f'결과:{pre}\n')
            gt = dataset.data.iloc[i]['answer']
            print(f'ground truth:{gt}\n')
            
            
            
    dataset.data = dataset.data.drop(columns=['image'])  
    output_path = '/workspace/last/dataset.xlsx'
    dataset.data.to_excel(output_path, index=False)

    judge_kwargs = {
        'nproc': 4,
        'verbose': False,
        'retry': 3,
        'model': "gpt-4o-mini"}

    eval_results = dataset.evaluate('/workspace/last/dataset.xlsx', **judge_kwargs)
    if True:
        dataset.data['solution'] = output['solution']
        output_path = '/workspace/last/dataset_include_solution.xlsx'
        dataset.data.to_excel(output_path, index=False)
    print(eval_results)

        # output

    

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM, MllamaForConditionalGeneration
#  MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info

def get_args():
    args = EasyDict({
        'task_name': 'scibench',
        'file': 'thermo_standardized',
        'propose_method': 'qwen',  # choices: ['qwen', 'gpt', 'llamaV_o1', 'llama3', 'llava']
        'value_method': 'qwen',  # choices: ['gpt', 'glm', 'local']
        'mode': 'mcts',  # choices: ['cot', 'tot', 'mcts']
        'temperature': 0.7,
        'time_limit': None,
        'iteration_limit': 5,
        'roll_forward_steps': 2, #2단계 simulation
        'roll_branch': 2, #다음 step에서 몇개의 step을 생성할지
        'roll_policy': 'greedy',  # choices: ['random', 'greedy']
        'exploration_constant': 0.4,
        'end_gate': 9.0,  # End threshold
        'branch': 3,
        
        'inf': 0.8,
        'evaluate': 'mathvista',  # Empty string means no evaluation
        'alpha': 0.5,
        'visualize': False,  # visualization
        'use_case_prompt': False,  # Use sample prompts
        'use_reflection': 'simple',  # choices: ['simple', 'common']
        'low': 0.0,
        'high': 1.0,
        'algorithm': 'dfs',  # choices: ['dfs', 'bfs']
        'select_branch': 2,
        'max_depth': 8,
        'select_method': 'greedy',  # choices: ['greedy', 'sample']
        'consistency': True,
        'model': '',
        'dataset': 'MathVista_MINI',
        'judge_args': None,
        'llamaV_o1': None,
        'Qwen2_5': "Qwen/Qwen2.5-VL-3B-Instruct",
        'llama3_vision_11b_model_path': None,
        'llava_next_8b_model_path': None,
        'openai_api_key': None,
        'openai_base_url': 'https://api.openai.com/v1',
        'dataset' :  'MathVista_MINI',
        'clip_model_path': 'openai/clip-vit-large-patch14-336'
        # --gpt_version 'gpt-4o' \


    })
    return args

# Example usage
args = get_args()
print(args.task_name)  # 'scibench'

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    run(args)
  
