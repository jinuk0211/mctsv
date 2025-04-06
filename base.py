from model import qwen, LLM, llama
from vlmeval.dataset import build_dataset
import torch
import os
import os.path as osp
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from easydict import EasyDict

def run():
    os.environ["OPENAI_API_KEY"] = 
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1/chat/completions" # Replace with your actual base
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")    
    print('-'*30, 'Begin testing', '-'*30, '\n')
    try:
        dataset_kwargs = {}
        dataset_name ="MathVista_MINI"
        dataset = build_dataset(dataset_name, **dataset_kwargs)
        print(f'전체 데이터셋 길이:{len(dataset.data)}')
        dataset.data = dataset.data.iloc[:6]
        data_len = len(dataset.data)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    model, processor = llama('llama')
    # model, processor = qwen('Qwen2_5')
    # llm, tokenizer, model_dict = LLM('qwen')
    output_list = []
    correct_count = 0
    for i in range(len(dataset.data)):
        image = dataset.data.iloc[i]['image']
        dataset.dump_image(dataset.data.iloc[i])
        img = osp.join(dataset.img_root, f"{dataset.data.iloc[i]['index']}.jpg")
        question = dataset.data.iloc[i]['question']
        if not pd.isnull(dataset.data.iloc[i]['choices']):
            choices = dataset.data.iloc[i]['choices'] 
        print('question')
        print(dataset.data.iloc[i]['question'])
        print('ground_truth:')      
        print(dataset.data.iloc[i]['answer'])
        response = get_proposal(model,processor,dataset.data.iloc[i]['question'],img,model_name ='qwen')
        
        print(f'\nresponse: {response}')
        print('\n=================================\n')

def get_proposal(model, processor, prompt, img_path, model_name ='llama'):
    #temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,do_sample=True, max_new_tokens=1024
    if model_name == 'qwen':
        response = []
        cnt = 2
        while not response and cnt:
            cnt -= 1
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        { "role": "user", 
                        "content": [{"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": f"{prompt}"},],
                        }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if not response:
            print(f'obtain<qwen>response fail!\n')
            return []
        return response[0]

    elif model_name == 'llama':

        image = Image.open(img_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},{"type": "text","text": f"{prompt}"}]}
                ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=500)
        print(processor.decode(output[0]))
        return output[0]    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    run()
#python base.py