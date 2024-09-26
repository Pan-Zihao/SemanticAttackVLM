import argparse
import openai
import json
import torch
from diffusers import FluxPipeline
from transformers import CLIPProcessor, CLIPModel
from prepareAEL import *


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


prompt = "Requirements: Generate an imaginative image description sentence according to the sentence format I give below. The more varied your descriptions are, the better. You need to use your imagination to the fullest, from elements of human society to the natural world, as well as science fiction or mythology. You can connect things that have no connection in the real world. Don't consider the constraints of reality. Format: a <picture/photo/watercolor/sketch> of a/an <color> <object> <appearance> in the style of <style>. <It/He/She> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <object> can be filled in with people, animals or any object, and <appearance> can be filled in with appearance descriptions such as wearing glasses. You can add descriptive sentences as appropriate. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."

def create_seed_json(code_list, objective_list, output_file):
    # 确保两个列表的长度相同
    if len(code_list) != len(objective_list):
        raise ValueError("code_list和objective_list的长度必须相同")

    # 构建JSON数据结构
    seed_data = []
    for code, objective in zip(code_list, objective_list):
        entry = {
            "algorithm": "",
            "code": code,
            "objective": objective,
            "other_inf": None
        }
        seed_data.append(entry)

    # 将数据写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(seed_data, file, ensure_ascii=False, indent=4)

def initialize(seed_number):
    for i in range(seed_number):
        caption = get_response(prompt)
        print(caption)
        code_list.append(caption)
        # 打开文件并写入字符串
        with open(args.captionfilename, "w", encoding="utf-8") as file:
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
        inputs = processor(text=caption, images=image_v, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        fitness = outputs.logits_per_image[0]
        print(fitness)
        objective_list.append(fitness)


if __name__ == "__main__":
    #Adding necessary input arguments
    parser = argparse.ArgumentParser(description='generate_seeds_captions')
    parser.add_argument('--seed_number',default=10, type=int)
    parser.add_argument('--captionfilename',default = "./seedfile/seedcaption.txt", type=str)
    parser.add_argument('--output_file',default='ael_seeds/seeds.json')

    args = parser.parse_args()
    code_list = []
    objective_list = []
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 设置设备为GPU
    else:
        device = torch.device("cpu")  # 如果CUDA不可用，使用CPU

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    model = CLIPModel.from_pretrained("clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("clip-vit-base-patch16")

    initialize(args.seed_number)

    # 调用函数生成seed.json文件
    create_seed_json(code_list, objective_list, args.output_file)
    print('generate seeds success!')



