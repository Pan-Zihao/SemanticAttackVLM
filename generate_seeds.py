import argparse
import os
import openai
import json
import torch
from PIL import Image
from diffusers import FluxPipeline
from x_flux.src.flux.xflux_pipeline import XFluxPipeline
from evaluation.VLMevaluation import caption_score, VQA_score
import shutil
import requests
from tqdm import trange

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

def create_seed_json(code_list, objective_list, output_file):
    # 确保两个列表的长度相同
    if len(code_list) != len(objective_list):
        raise ValueError("code_list和objective_list的长度必须相同")

    # 构建JSON数据结构
    seed_data = []
    for code, objective in zip(code_list, objective_list):
        if isinstance(objective, torch.Tensor):
            if objective.numel() == 1:  # to scalar or list
                objective = objective.item()  
            else:
                objective = objective.tolist()  
        entry = {
            "algorithm": "",
            "code": code,
            "objective": objective,
            "other_inf": None
        }
        seed_data.append(entry)
        print("code",code,"objective", objective)

    # 将数据写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(seed_data, file, ensure_ascii=False, indent=4)


def initialize(QA_number, prompt, VLMpath, pattern):
    for i in trange(QA_number):
        generated_prompts = get_response(prompt)
        generated_prompts_list = [prompt.strip() for prompt in generated_prompts.split('\n') if prompt.strip()]
        code_list.extend(generated_prompts_list)
    print(f"=============total generate {len(code_list)} prompts ============")


    with open(args.captionfilename, "w", encoding="utf-8") as file:
        for i in trange(len(code_list)):
            caption = code_list[i]
            print(caption)
            # 打开文件并写入字符串
            file.write(caption + "\n")  # 写入字符串并在每个字符串后添加换行符
            if args.stage == 1:
                image = pipe(
                    caption,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.stage == 2:
                image = xflux_pipeline(
                    prompt=caption,
                    width=1024,
                    height=1024,
                    guidance=4,
                    num_steps=25,
                    seed=i,
                    true_gs=3.5,
                    timestep_to_start_cfg=5,
                    image_prompt=image_prompt,
                    ip_scale=1.0,
                )
            image_path = os.path.join(args.imagefilename, f"{i}.png")
            image.save(image_path)
            if pattern == 'caption':
                fitness = caption_score(caption, VLMpath, image_path)
            elif pattern == 'VQA':
                fitness = VQA_score(caption, VLMpath, image_path)
            print(fitness)
            objective_list.append(fitness)



if __name__ == "__main__":
    #Adding necessary input arguments
    parser = argparse.ArgumentParser(description='generate_seeds_captions')
    parser.add_argument('--QA_number',default=2, type=int)
    parser.add_argument('--captionfilename',default = "./seedfile1/seedcaption.txt", type=str)
    parser.add_argument('--imagefilename',default = "./seedfile1/seedimage", type=str)
    parser.add_argument('--output_file',default='ael_seeds/seeds.json')
    parser.add_argument('--stage',default=1, type=int)
    # 使用示例
    # model_path = "model/llava-v1.5-7b"
    # model_path = "model/llava-v1.5-13b"
    # model_path = "model/llava-v1.6-vicuna-7b"
    # model_path = "model/llava-v1.6-vicuna-13b"
    parser.add_argument('--VLMpath', default = "/storage/panzihao/models/llava-v1.5-7b", type=str)
    parser.add_argument('--pattern', default="caption", type=str)
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default='/storage/panzihao/models/flux-ip-adapter',
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )


    args = parser.parse_args()
    code_list = []
    objective_list = []
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 设置设备为GPU
    else:
        device = torch.device("cpu")  # 如果CUDA不可用，使用CPU
    if os.path.exists(args.captionfilename):
        os.remove(args.captionfilename)
    if os.path.exists(args.imagefilename):
        shutil.rmtree(args.imagefilename)
        os.mkdir(args.imagefilename)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    if args.stage == 1:
        prompt = "Requirements: Generate lots of image description sentences according to the sentence format I give below, separeted by '\n'.Do not add any numbering or bullets, strictly follow the instructions. Format: a <picture/photo/watercolor/sketch> of a/an <color> <object> <appearance> in the style of <style>. <It/He/She> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <object> can be filled in with people, animals or object, and <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
        pipe = FluxPipeline.from_pretrained("/storage/panzihao/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        
        initialize(args.QA_number, prompt, args.VLMpath, args.pattern)

        with open(args.captionfilename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 过滤掉空行，并计算非空行的数量
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        # 计算非空行的数量
        seed_number = len(non_empty_lines)

        # 检查文件是否存在
        if os.path.exists(args.captionfilename):
            # 删除文件
            os.remove(args.captionfilename)
            print(f"File {args.captionfilename} has been deleted successfully.")
        else:
            print(f"File {args.captionfilename} does not exist.")

        # 将结果写入新文件
        with open(args.captionfilename, 'w', encoding='utf-8') as file:
            file.writelines(non_empty_lines)

        # 调用函数生成seed.json文件
        create_seed_json(code_list, objective_list, args.output_file)
        print('generate seeds success!')
        print(f'the number of seeds is {seed_number}')
    elif args.stage == 2:
        with open('results1.json', 'r') as f1:
            results1 = json.load(f1)
        with open('objective1.json', 'r') as f2:
            objective1 = json.load(f2)
        min_key = min(objective1, key=lambda k: objective1[k])
        image_prompt = Image.open(results1[min_key])    
        object = get_response(f"Give the subject of the sentence. No adjectives or other words are needed. Just follow the instructions and give a word.\n {min_key}")
        if os.path.exists('./object.txt'):
            os.remove('./object.txt')
        with open('./object.txt','w',encoding='utf-8') as file:
            file.write(object)
        prompt = f"Requirements: Generate lots of image description sentences according to the sentence format I give below, separeted by '\n'.Do not add any numbering or bullets, strictly follow the instructions. Format: a <picture/photo/watercolor/sketch> of a/an <color> {object} <appearance> in the style of <style>. <It/He/She> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: {min_key}"
        xflux_pipeline = XFluxPipeline(args.model_type, device, args.offload)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
        initialize(args.QA_number, prompt, args.VLMpath, args.pattern)
        with open(args.captionfilename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 过滤掉空行，并计算非空行的数量
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        # 计算非空行的数量
        seed_number = len(non_empty_lines)

        # 检查文件是否存在
        if os.path.exists(args.captionfilename):
            # 删除文件
            os.remove(args.captionfilename)
            print(f"File {args.captionfilename} has been deleted successfully.")
        else:
            print(f"File {args.captionfilename} does not exist.")

        # 将结果写入新文件
        with open(args.captionfilename, 'w', encoding='utf-8') as file:
            file.writelines(non_empty_lines)

        # 调用函数生成seed.json文件
        create_seed_json(code_list, objective_list, args.output_file)
        print('generate seeds success!')
        print(f'the number of seeds is {seed_number}')




