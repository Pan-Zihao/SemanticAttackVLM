import argparse
import sys
import os
import json
import shutil
import re
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
sys.path.append(ROOT_PATH)  # This is for finding all the modules
from aell.src.aell import ael
from aell.src.aell.utils import createFolders
import torch
from diffusers import FluxPipeline
from transformers import CLIPProcessor, CLIPModel
from prepareAEL import *

def get_output(file_path='./ael_results/combined_results.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    code_list = [entry["code"] for entry in data.values()]
    caption_path = './stage1_results/captions/captions.txt'
    os.makedirs(os.path.dirname(caption_path), exist_ok=True)

    with open(caption_path, 'w', encoding='utf-8') as file:
        for caption in code_list:
            file.write(caption + "\n")

    return code_list


def copy_top_K_images(source_folder, target_folder, K):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有图片文件
    images = [f for f in os.listdir(source_folder) if re.match(r'.*\.png', f)]

    # 如果文件夹为空或图片不足K张，返回错误
    if len(images) == 0:
        print("No images found in the source folder.")
        return
    if len(images) < K:
        print(f"Only {len(images)} images found, not enough to copy K.")
        return

    # 根据索引排序图片
    images.sort(key=lambda x: int(re.search(r'(\d+)', x).group()), reverse=True)

    # 选择索引最大的K张图片
    top_K_images = images[:K]

    # 复制图片到新文件夹
    for image in top_K_images:
        source_path = os.path.join(source_folder, image)
        target_path = os.path.join(target_folder, image)
        shutil.copy(source_path, target_path)
        print(f"Copied {image} to {target_folder}")

### Debug model ###
debug_mode = False# if debug

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description='main')
    # number of algorithms in each population, default = 10
    parser.add_argument('--pop_size', default=2, type=int)
    # number of populations, default = 10
    parser.add_argument('--n_pop',default=2,type=int)
    # number of parents for 'e1' and 'e2' operators, default = 2
    parser.add_argument('--m',default=2,type=int)
    parser.add_argument('--source_folder', default="./images_stage1", type=str)
    parser.add_argument('--target_folder', default="./stage1_results/images")
    parser.add_argument('--object',default=False,type=bool)

    args = parser.parse_args()

    ### LLM settings  ###

    use_local_llm = False  # if use local model
    url = None  # your local server 'http://127.0.0.1:11012/completions'

    api_endpoint = "oa.api2d.site"
    # api_key = "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" # use your key

    api_key = "sk-ePaBZR3FUIwaQNojF0871e9a338d44C5B4D332B8B6B8968e"
    # llm_model = "gpt-3.5-turbo-1106"
    llm_model = "gpt-4o"
    ### output path ###
    output_path = "./"  # default folder for ael outputs
    createFolders.create_folders(output_path)
    load_data = {
        'use_seed': True,
        'seed_path': output_path + "ael_seeds/seeds.json",
        "use_pop": False,
        "pop_path": output_path + "ael_results/pops/population_generation_0.json",
        "n_pop_initial": 0
    }
    ### Debug model ###
    debug_mode = False  # if debug

    # AEL
    operators = ['e1','m1']  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
    operator_weights = [1,1] # weights for operators, i.e., the probability of use the operator in each iteration , default = [1,1,1,1]

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 设置设备为GPU
    else:
        device = torch.device("cpu")  # 如果CUDA不可用，使用CPU

    pipe = FluxPipeline.from_pretrained("/storage/panzihao/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    model = CLIPModel.from_pretrained("/storage/panzihao/models/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/storage/panzihao/models/clip-vit-large-patch14")
    eva = Evaluation(pipe,model,processor,device=device)

    print(">>> Start AEL ")
    if not os.path.exists(args.source_folder):
        os.makedirs(args.source_folder)
    algorithmEvolution = ael.AEL(use_local_llm, url,
        api_endpoint,api_key,llm_model,args.pop_size,args.n_pop,
        operators,args.m,operator_weights,load_data,output_path,debug_mode,evaluation=eva)

    # run AEL
    algorithmEvolution.run(object=args.object)

    print("AEL successfully finished !")
    print(get_output())
    # 调用函数
    copy_top_K_images(args.source_folder, args.target_folder, K=args.n_pop)



