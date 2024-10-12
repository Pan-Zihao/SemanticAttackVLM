import random
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
import os
import json
import requests
nltk.download('punkt')
nltk.download('wordnet')



              
def calculate_meteor(reference_sentences, hypothesis_sentence):
    reference_sentences = [reference_sentences]
    lemmatizer = WordNetLemmatizer()

    reference_tokens = [word_tokenize(ref) for ref in reference_sentences]
    hypothesis_tokens = word_tokenize(hypothesis_sentence)

    reference_tokens = [[lemmatizer.lemmatize(word) for word in ref] for ref in reference_tokens]
    hypothesis_tokens = [lemmatizer.lemmatize(word) for word in hypothesis_tokens]

    score = meteor_score.single_meteor_score(reference_tokens[0], hypothesis_tokens)
    return score

def calculate_BLEU(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    bleu_score = sentence_bleu([reference], candidate)
    return bleu_score

def calculate_rouge(reference_summary, generated_summary):
    rouge = Rouge()

    # 计算 ROUGE 分数
    scores = rouge.get_scores(generated_summary, reference_summary)[0]

    # 计算平均 F1 分数
    average_f1 = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    return average_f1


def image_caption(model_path, image_file, prompt):
    load_model_once(model_path)

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    result = eval_model(args)
    return result


tokenizer = None
model = None
image_processor = None
context_len = None
# 模型加载
def load_model_once(model_path):
    global tokenizer, model, image_processor, context_len
    if model is None:  # 如果模型未加载，则加载
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

def evaluate_image(model_path, image_file, prompt):
    # 加载模型（只在第一次调用时加载）
    load_model_once(model_path)

    # 构造参数
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # 评估模型并返回结果
    result = eval_model(args)
    return result

def get_response(user_message):
    Baseurl = "https://api.claude-Plus.top"
    Skey = "sk-vjulMaFmBWm31NP4OqwnKaDJMb3X0jbVlnIvg4XbYgtXwzWi"

    payload = json.dumps({
        "model": "claude-3-5-sonnet-20240620",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    })

    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)

    data = response.json()

    content = data['choices'][0]['message']['content']

    return content

# 使用示例
#model_path = "model/llava-v1.5-7b"
#model_path = "model/llava-v1.5-13b"
#model_path = "model/llava-v1.6-vicuna-7b"
#model_path = "model/llava-v1.6-vicuna-13b"
#image_file = "鸟人.png"
#prompt = "Please describe the image in the following format: a (picture/photo/watercolor/sketch) of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture> on the <background describe> in the <location> on a <weather> day, <action description>, <environment description>."

#result = image_caption(model_path, image_file, prompt)
#print(result)


#TODO
def caption_score(caption, model_path, image_file):
    prompt = "Please describe the image in the following format: a (picture/photo/watercolor/sketch) of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture> on the <background describe> in the <location> on a <weather> day, <action description>, <environment description>."
    pre_caption = image_caption(model_path, image_file, prompt)
    #score = random.random()
    score = calculate_rouge(caption,pre_caption)+calculate_meteor(caption,pre_caption)+calculate_BLEU(caption,pre_caption)
    return score


def VQA_score(gt,model_path,image_file):
    prompt = []
    prompt.append("Please describe the artistic style of the image, including but not limited to impressionism, cyberpunk, nouveau, etc. Please answer in the format: 'This image is in the xxx style.' You do not need to explain.")
    prompt.append("Please describe the main subject of the image. Please answer in the format: 'The main subject of this image is xxx.' You do not need to explain.")
    prompt.append("Please describe the number of subjects in the image. Please answer in the format: 'The number of subjects in this image is xxx.' You do not need to explain.")
    prompt.append("Please describe the color of the subject in the image. Please answer in the format: 'The color of the subject in this image is xxx.' You do not need to explain.")
    prompt.append("Please describe the background of the image in detail, including but not limited to the objects in the background. Please answer in the format: 'The background of this image contains xxx.' You do not need to explain.")
    prompt.append("Please describe the weather in the image. Please answer in the format: 'The weather in this image is xxx.' You do not need to explain.")
    result = []
    for i in range(len(prompt)):
        result.append(evaluate_image(model_path, image_file, prompt[i]))
        print(result[i])
    gt_prompt = []
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the artistic style of the image, including but not limited to impressionism, cyberpunk, nouveau, etc. Please answer in the format: 'This image is in the xxx style.' You do not need to explain.")
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the main subject of the image. Please answer in the format: 'The main subject of this image is xxx.' You do not need to explain.")
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the number of subjects in the image. Please answer in the format: 'The number of subjects in this image is xxx.' You do not need to explain.")
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the color of the subject in the image. Please answer in the format: 'The color of the subject in this image is xxx.' You do not need to explain.")
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the background of the image in detail, including but not limited to the objects in the background. Please answer in the format: 'The background of this image contains xxx.' You do not need to explain.")
    gt_prompt.append("This is a description of an image: "+gt+" Please answer the weather in the image. Please answer in the format: 'The weather in this image is xxx.' You do not need to explain.")
    score = []
    gt_result = []
   
    for i in range(len(gt_prompt)):
        gt_result.append(get_response(gt_prompt[i]))
        
    for i in range(len(gt_result)):
        score.append(calculate_rouge(gt_result[i],result[i])+calculate_meteor(gt_result[i],result[i])+calculate_BLEU(gt_result[i],result[i]))
    
    average_score = sum(score) / len(score)
    return average_score

#example
#model_path = "model/llava-v1.5-7b"
#image_file = "鸟人.png"
#gt = "A parrot in the style of an oil painting."
#VQA_score(gt,model_path,image_file)
