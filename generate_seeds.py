import argparse
import os
import openai
import json
import torch
from PIL import Image
from diffusers import FluxPipeline
from transformers import CLIPProcessor, CLIPModel
from ipadapter_x_flux.src.flux.xflux_pipeline import XFluxPipeline

def get_response(prompt_content):
    BASE_URL = "https://api.xiaoai.plus/v1"
    OPENAI_API_KEY = "sk-ePaBZR3FUIwaQNojF0871e9a338d44C5B4D332B8B6B8968e"
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "user", "content": prompt_content}
        ]
    )

    return response.choices[0].message.content

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

def initialize(QA_number):
    with open(args.captionfilename, "w", encoding="utf-8") as file:
        for i in range(QA_number):
            caption = get_response(prompt)
            print(caption)
            code_list.append(caption)
            # 打开文件并写入字符串
            file.write(caption + "\n")  # 写入字符串并在每个字符串后添加换行符
            image = pipe(
                caption,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            image.save(f"./seedfile/seedimage/{i}.png")
            image_v = Image.open(f"./seedfile/seedimage/{i}.png")
            inputs = processor(text=caption, images=image_v, return_tensors="pt", padding=True, truncation=True,max_length=77).to(device)
            outputs = model(**inputs)
            fitness = outputs.logits_per_image[0]
            print(fitness)
            objective_list.append(fitness)


if __name__ == "__main__":
    #Adding necessary input arguments
    parser = argparse.ArgumentParser(description='generate_seeds_captions')
    parser.add_argument('--QA_number',default=10, type=int)
    parser.add_argument('--captionfilename',default = "./seedfile/seedcaption.txt", type=str)
    parser.add_argument('--output_file',default='ael_seeds/seeds.json')
    parser.add_argument('--stage',default=1, type=str)
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )

    args = parser.parse_args()
    code_list = []
    objective_list = []
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 设置设备为GPU
    else:
        device = torch.device("cpu")  # 如果CUDA不可用，使用CPU

    if args.stage == 1:
        prompt = "Requirements: Generate some imaginative image description sentences according to the sentence format I give below. The content can involve science fiction, art, realism, fantasy, etc. Format: a <picture/photo/watercolor/sketch> of <color> <object> <appearance> in the style of <style>. <It/He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <object> can be filled in with people, animals or any object, and <appearance> can be filled in with appearance descriptions such as wearing glasses. You can add descriptive sentences as appropriate. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
        pipe = FluxPipeline.from_pretrained("/storage/panzihao/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        model = CLIPModel.from_pretrained("/storage/panzihao/models/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("/storage/panzihao/models/clip-vit-large-patch14")
        if os.path.exists(args.captionfilename):
            os.remove(args.captionfilename)
        initialize(args.QA_number)

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
    else:
        prompt = ""
        xflux_pipeline = XFluxPipeline(args.model_type, device, args.offload)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

        images_stage1_path = "stage1_results/images"
        captions_stage1_path = "stage1_results/captions"
        images_stage1 = []
        captions_stage1 = []


        for root, dirs, files in os.walk(images_stage1_path):
            for file in files:
                if file.lower().endswith(tuple(image_extensions)):
                    images_stage1.append(os.path.join(root, file))

        with open(captions_stage1_path, 'r', encoding='utf-8') as file:
            for line in file:
                captions_stage1.append(line.strip())

        keys_tuple = tuple(images_stage1)
        stage1_results = dict(zip(keys_tuple, captions_stage1))
        for i in stage1_results.keys():
            image_prompt = Image.open(args.img_prompt)



