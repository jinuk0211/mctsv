from mcts import MCTS
from searchtask import SearchTask
from model import get_proposal, llm_proposal
from utils import extract_summary_from_solution, llm_verify, exact_match_score
from transformers import CLIPModel, AutoProcessor,Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14-336"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from prompt import llm_prompt
import re

# Task = MCTS_Task(question, model, processor, args.propose_method, args.value_method, args.branch, args.end_gate,
#                     args.roll_policy, args.roll_branch, args.roll_forward_steps, args.time_limit,
#                     args.iteration_limit, args.exploration_constant, args.alpha, args.inf,
#                     args.temperature, use_case_prompt=args.use_case_prompt, use_reflection=args.use_reflection,
#                     low=args.low, high=args.high, evaluate=args.evaluate,img_path=img)



class MCTS_Task(SearchTask):
    def __init__(self,data, model, processor, propose_method='qwen', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=3, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='en', weighted_verify=False, img_path='',model_dict=None):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.model = model
        self.processor = processor
        self.img_path = img_path
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = 3
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = 1
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify
        self.model_dict = model_dict
    def update_count(self):
        self.node_count += 1
        
    def set_limit_type(self):
      if self.time_limit is not None:
          if self.iteration_limit is not None:
              raise ValueError("Cannot have both a time limit and an iteration limit")
          # time taken for each MCTS search in milliseconds
          self.limit_type = 'time'
      else:
          if self.iteration_limit is None:
              raise ValueError("Must have either a time limit or an iteration limit")
          # number of iterations of the search
          if self.iteration_limit < 1:
              raise ValueError("Iteration limit must be greater than one")
          self.limit_type = 'iterations'

    def run(self):
        self.clear_cache()
        self.set_limit_type()
    
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':#value_model
            if self.sample_value != 'full':
                if self.evaluate == 'mathvista':  # SciBench style
                    solution = node.y
                    #summary = self.get_summary(solution)
                    if self.lang == 'en':
                        #summary 제공
                        prompt = self.MATH_summary_prompt_wrap(self.question, solution)
                        response = get_proposal(self.model,self.processor,prompt, self.img_path)

                        if not response:
                            print('Failed to get the review!\n')
                            return ''
                        # p = ''
                        # for _ in response:
                        #     p = p + _
                        # summ = p.strip()
                        print(f'math_summary_prompt에 의한 결과:{response}\n')
                        summary =response.split("The final answer is")[-1].strip()
                            #get_summary 끝부분
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
                        final_answer.update({'value_samples': new_value_samples})
                else:  # MATH style self.evaluate == 'scibench"의 else문
                    solution = node.y
                    cnt = 5
                    summ = ''
                    while cnt:
                        if self.verify_method == 'string':
                            summ = self.get_MATH_summary(solution)
                        else:
                            summ = self.get_summary(solution)
                        if summ:
                            node.summary = summ
                            break
                        else:
                            cnt -= 1

                    if not summ:
                        summ = extract_summary_from_solution(solution)
                        node.summary = summ

                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'real_answer': self.answer}
                return final_answer, root
            
    def get_first_step(self, y):
        prompt = self.image_description(self.question, y)
        response = get_proposal(self.model, self.processor, prompt,self.img_path)
        print(f'처음 생성된 응답:{response}\n')
        return 'Image Description: ' + response + '\n'
    
    def get_next_step(self, y, step_n):

        if self.propose_method == 'gpt':
            prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response =  get_proposal(self.model, self.processor, prompt, self.img_path)
        if not response:
            print('Failed to get the next step!\n')
            return ''
        print(f'response:{response}')
        # if len(response) > 5:
        #     response = response[:5]
        
        if response.startswith("Next step: "):  
            stp = response[len("Next step: "):]  # "Next step: " 길이만큼 잘라냄
            revised_ = 'Step ' + str(step_n-1) + ': ' + stp
        else:
            stp = response  # "Next step: "이 없으면 그대로 유지
            revised_ = stp
        print(f'revised 이후의 step: {revised_}\n')
        return revised_ + '\n'



    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
        else:
            reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        try:
          cnt = 3
          response = []
          while not response and cnt:
            #   response = get_proposal(self.model, self.processor, reflection_prompt,self.img_path)
              response = llm_proposal(self.model_dict['model'], self.model_dict['tokenizer'], reflection_prompt)
              cnt -= 1
                  
        except Exception as e:
          print(f'obtain<{self.propose_method}>reflection fail!\nError:{e}\n')
          return ''
        # if not response:
        #     print('获得意见失败！\n')
        #     return '<end>'
        print(f'reflection 결과:{response}')


       
        
        if 'unsolved' in response or step_n <= 1:
            print('revised된 reflection: <continue>\n')
            return '<continue>'
        elif 'solved' in response:
            print('revised된 reflection: <end>\n')
            return '<end>'
        else:
            print('revised된 reflection: <continue>\n')
            return '<continue>'

    def get_reflection(self, y, step_n):
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.lang == 'zh':
            if self.propose_method == 'gpt' or self.propose_method == 'local':
                reflection_prompt = self.single_reflection_wrap_gpt(self.question, y, step_n)
            elif self.propose_method == 'llama':
                reflection_prompt = self.single_reflection_wrap_llama(self.question, y, step_n)
            else:
                reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)
        else:
            reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(self.model, self.processor, reflection_prompt,self.img_path)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return ''

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

       
 
        if 'Problem solved' in p:
            print('revised된 reflection: <end>\n')
            return '<end>'
        else:
            if 'Analysis:' not in p:
                print('输出格式有误！\n')
                return ''
            revised_ = p.split('Analysis:')[1].strip()
            print(f'revised된 reflection:{revised_}\n')
            return revised_


    def get_summary(self, y):
        if self.lang == 'zh':
            if self.evaluate == 'scibench':
                prompt = self.evaluate_summary_prompt_wrap(self.question, y)
            elif self.evaluate == 'scieval':
                prompt = self.general_evaluate_summary_prompt_wrap(self.question, y)
            else:
                prompt = self.summary_prompt_wrap(self.question, y)

            response = get_proposal(self.model, self.processor, prompt, self.img_path)

            if not response:
                print('Failed to get the review!失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            p = p.strip()

            if self.evaluate:
                if len(p) < 1:
                    print('Failed to get the review!过短！\n')
                    return ''

                if '综上所述，最终答案是:' not in p:
                    summ = '综上所述，最终答案是:' + p
                    print(f'Failed to get the review!:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                    print(f'Failed to get the review!:{summ}\n')
                    return summ

            else:
                if len(p) < 1:
                    print('Failed to get the review!过短！\n')
                    return ''

                p = p.replace('综上所述,', '综上所述，')
                if '综上所述，' not in p:
                    summ = '综上所述，' + p
                    print(f'Failed to get the review!:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，' + p.split('综上所述，')[-1]
                    print(f'Failed to get the review!:{summ}\n')
                    return summ

        else:
            prompt = self.MATH_summary_prompt_wrap(self.question, y)
            response = get_proposal(self.model, self.processor, prompt,self.img_path)
            if not response:
                print('Failed to get the review!失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _
            summ = p.strip()
            print(f'Failed to get the review!:{summ}\n')

            return summ

    def get_MATH_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(self.model, self.processor, prompt, self.img_path)
        if not response:
            print('Failed to get the review!失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'Failed to get the review!:{p}\n')
        return p

    def verify_end_nodes(self, root):
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()
        flag = False
        for leaf in end_leaf_nodes:
            leaf.on_final_route = True
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(leaf.y)
                else:
                    summ = self.get_summary(leaf.y)
                if summ:
                    leaf.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(leaf.y)
                leaf.summary = summ

            if self.verify_method == 'string':
                result = exact_match_score(summ, self.answer)
            else:
                result = llm_verify(summ, self.answer)
            if result:
                if self.reward_model_type == 'vm':
                    leaf.min_steps_to_correct = 1
                else:
                    leaf.he = 1
                flag = True
        return flag, end_leaf_nodes


        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()

        if not end_leaf_nodes or not weighted:
            if not end_leaf_nodes:
                best_node, best_V = root.getBestV()
            else:
                sorted_nodes = sorted(end_leaf_nodes, key=lambda x: x.V, reverse=True)
                best_node = sorted_nodes[0]
            solution = best_node.y
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(solution)
                else:
                    summ = self.get_summary(solution)
                if summ:
                    best_node.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(solution)
                best_node.summary = summ
            return solution, summ

        else:
            all_answers = {}  # {answer: [solution, summ, value]}
            for leaf in end_leaf_nodes:
                cnt = 5
                summ = ''
                while cnt:
                    if self.verify_method == 'string':
                        summ = self.get_MATH_summary(leaf.y)
                    else:
                        summ = self.get_summary(leaf.y)
                    if summ:
                        leaf.summary = summ
                        break
                    else:
                        cnt -= 1
                if not summ:
                    summ = extract_summary_from_solution(leaf.y)
                    leaf.summary = summ

                extracted_answer = extract_answer(summ)
                if extracted_answer in all_answers.keys():
                    all_answers[extracted_answer][2] += leaf.V
                else:
                    all_answers[extracted_answer] = [leaf.y, summ, leaf.V]

            best_answer = max(all_answers.values(), key=lambda x: x[2])
            solution = best_answer[0]
            summ = best_answer[1]
            return solution, summ

    def get_step_value(self, y, action):
        print(' get step value 함수 시작\n')
        print(f'y:{y}\n')
        # print(f'action:{action}')
        if y in self.value_cache.keys():
            print('캐시 value 사용됨\n')
            return self.value_cache[y]
        prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y
        if 'Image Description' in action:
            print('image description에 대한 value 생성중')
            lmm_prompt = self.image_description_score(self.question, y)
            response = get_value(self.model,self.processor, prompt_answer, lmm_prompt, action, self.value_method, img_path=self.img_path)
            value = self.value_outputs_unwrap(response, self.low, 10.0)
            # value = (1-self.alpha)*confidence + self.alpha*value
            print(f'unwrap된 value:{value}\n') #평가받기
            print(' get step value 함수 끝\n')
            # from model import get_clip_score
            # clip_score = get_clip_score(action,self.img_path,model_dict['clip'],model_dict['clip_processor'])
            #   def get_clip_score(new_text, image, model, processor):
            self.value_cache.update({y: value})
            # return llm_value* 0.5 + value * 0.5
            return value
        else:
            lmm_prompt = self.value_prompt_wrap(self.question, y) 
            llm_prompt = self.llm_prompt(self.question, y)
            response = get_value(self.model,self.processor, prompt_answer, lmm_prompt, action, self.value_method, img_path=self.img_path)
            value = self.value_outputs_unwrap(response, self.low, 10.0)
        # from model import llm_proposal
        # llm_response = llm_proposal(self.model_dict['model'],self.model_dict['tokenizer'],llm_prompt)
        # llm_value = self.value_outputs_unwrap(llm_response, self.low, 10.0)    
        # print(f'unwrap된 llm value:{llm_value}\n')                
            print(f'unwrap된 value:{value}\n') #평가받기
            print(' get step value 함수 끝\n')
            self.value_cache.update({y: value})
            # return llm_value* 0.5 + value * 0.5
            return value            
        # if self.value_method == 'local_prm': #clip, prm + llm + prm
        #     confidence, value_score = get_value(self.model,self.processor, prompt_answer, llm_prompt, lmm_prompt, action, self.value_method, img_path=self.img_path)
        #     value = (1-self.alpha)*confidence + self.alpha*value_score
        #     print(f'value:{value}\n') #评分
        #     self.value_cache.update({y: value})
        #     return value


  
# clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')
# clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')


def get_value(model, processor, prompt, lmm_prompt, action, value_method, img_path):
    response = []
    cnt = 2
    if 'Image Description' in action:
        # clip_score = get_clip_score(action,img_path,clip_model,clip_processor)
        response = get_proposal(model, processor, lmm_prompt, img_path)
        # return clip_score, response
        print(f'value로 생성된 값:{response}')
        return response
        
    else:  #lmm은 둘다동일하지만 이제 img -> clip, reasoning_step -> llm
        while not response and cnt:
            # value = LLM(llm_prompt, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
            response = get_proposal(model, processor, lmm_prompt, img_path)

            cnt -= 1
        # if not value:
        #     print(f'obtain<{method}>score fail!\n')
        #     return []
        # return value, response
        print(f'value로 생성된 값:{response}')
        return response


#utils.py
    #MCTS_Task 클래스
