from prompt import *
import re
import os

class SearchTask(object):
    def __init__(self, data, model, processor, propose_method='qwen', value_method='glm'):
        super().__init__()
        self.question = data
        self.model = model
        self.processor = processor
        # self.clip = clip
        # self.clip_processor = clip_processor
        # self.llm = llm
        # self.tokenizer = tokenizer
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def image_description_score(x,y):
        print('\n', '==============================', 'image description score', '==============================', '\n')
        # prompt = image_description_score + x + "\n" +y
        if "<|eot_id|>" in y:
            y = y.replace("<|eot_id|>", "")        
        prompt = image_description_score + "\n" +y
        return prompt
    @staticmethod
    def image_description(x,y):
        print('\n', '==============================', 'image description', '==============================', '\n')

        prompt = image_description_prompt 
        return prompt        
    @staticmethod
    def llm_prompt(x,y):

        prompt = llm_prompt + x + '\n' + y
        return prompt
    @staticmethod
    def summary_prompt_wrap(x: str, y: str = '') -> str:
        summary_prompt = '''
        Your task is to summarize the final answer in one sentence based on a given science/math problem and its completed solution steps, following the specified format.

        Given problem: '''
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\n' + y + 'The summary based on the reasoning steps i:\n')
        prompt = summary_prompt + x + '\n' + y + '\noutput:'
        return prompt

    @staticmethod
    def MATH_summary_prompt_wrap(x: str, y: str = '') -> str:
        MATH_summary_prompt = '''
        Given a problem, image, image description and its corresponding solution, your task is to extract the final answer obtained in the solution.
        You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.'''


        print('\n', '==============================', 'summary', '==============================', '\n')
        print('math_summary_prompt 대상:' + y + '\n\n')
        prompt = x + '\nSolution: ' + y + '\n' + MATH_summary_prompt +'\nExtracted answer:' 
        return prompt



    # @staticmethod
    # def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
    #     print('\n', '==============================', 'single_propose_prompt_wrap', '==============================', '\nstep: ', step)
    #     print('single_propose_prompt: \n', x + '\n' + y + '\n')
    #     if "<|eot_id|>" in y:
    #         y = y.replace("<|eot_id|>", "")             
    #     prompt = single_proposal_prompt + x + y + '\noutput:'
    #     return prompt


    @staticmethod
    def zero_single_propose_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'zero_single_propose_wrap', '==============================', '\nstep: ', step)
        print('프롬프트 \n 이전 solution:'+ y + '\n 프롬프트 끝\n\n')

        if "<|eot_id|>" in y:
            y = y.replace("<|eot_id|>", "")                   
        prompt = x + '\n' + y + '\n' + zero_single_proposal_prompt_en
        return prompt


 

   

    # @staticmethod
    # def single_reflection_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
    #     print('\n', '==============================', 'single_reflection_wrap', '==============================', '\nstep: ', step)
    #     print('reflection_prompt: \n', x + '\nexisting step:\n' + y + '基于以上步骤给出的意见:\n')

    #     if not y:
    #         y = 'None\n'
    #     prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
    #     return prompt


    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0, lang: str = 'en') -> str:
        print('\n', '==============================', 'single_reflection_wrap_simple', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n' + y + '이전 단계에 대한 reflection:\n')
        if lang == 'en':
            if not y:
                y = 'None\n'
            prompt = x + '\n' + y + single_reflection_prompt_simple_en 
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'value_prompt', '==============================', '\n')
        value_prompt = x + '\n' + y.strip() + critic_simplified
        return value_prompt



    @staticmethod
    def value_outputs_unwrap(value_outputs, low=0.0, high=10) -> float:
        out_value = low
        print(f'value_unwrap 안되는 이유:{value_outputs,type(value_outputs)}')
        if 'Score' not in value_outputs:
            print('점수 출력이 올바르지 않습니다 value_outputs_unwrap\n')
        try:        
            if "<|eot_id|>" in value_outputs:
                value_outputs = value_outputs.replace("<|eot_id|>", "")
            
            if '**Score:**' in value_outputs:
                number = re.search(r"[-+]?\d*\.?\d+", value_outputs)  # 정수 및 소수 지원
                if number:
                    out_value = number.group()
                    out_value = float(out_value)
            else: 
                out_value = float(value_outputs.split(":")[1].strip())
                out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'점수 출력에 오류가 있습니다! 오류 유형:{e}\n')
            return low
        print(f'최종:{out_value,type(out_value)}')

        return out_value   
