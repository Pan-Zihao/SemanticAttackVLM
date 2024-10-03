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
from x_flux.src.flux.xflux_pipeline import XFluxPipeline
from prepareAEL import *

def get_output(results, caption_path, image_path, file_path='./ael_results/combined_results.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    code_list = [entry["code"] for entry in data.values()]
    os.makedirs(os.path.dirname(caption_path), exist_ok=True)

    with open(caption_path, 'w', encoding='utf-8') as file:
        for caption in code_list:
            file.write(caption + "\n")
            image = results[caption]
            shutil.copy(image, image_path)

### Debug model ###
debug_mode = False# if debug

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description='main')
    # number of algorithms in each population, default = 10
    parser.add_argument('--pop_size', default=50, type=int)
    # number of populations, default = 10
    parser.add_argument('--n_pop',default=5,type=int)
    # number of parents for 'e1' and 'e2' operators, default = 2
    parser.add_argument('--m',default=4,type=int)
    parser.add_argument('--caption_path', default="./stage1_results/captions/captions.txt")
    parser.add_argument('--image_path', default="./stage1_results/images")
    parser.add_argument('--stage', default=1, type=int)
    # 使用示例
    # model_path = "model/llava-v1.5-7b"
    # model_path = "model/llava-v1.5-13b"
    # model_path = "model/llava-v1.6-vicuna-7b"
    # model_path = "model/llava-v1.6-vicuna-13b"
    parser.add_argument('--VLMpath', default="model/llava-v1.5-7b", type=str)
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
        "--ip_local_path", type=str, default=None,
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
    if os.path.exists(args.caption_path):
        os.remove(args.caption_path)
    if os.path.exists(args.image_path):
        shutil.rmtree(args.image_path)
        os.mkdir(args.image_path)

    if args.stage == 1:
        pipe = FluxPipeline.from_pretrained("/storage/panzihao/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        eva = Evaluation(pipe,image_prompt=None,stage=1,pattern=args.pattern,VLMpath=args.VLMpath)
        print(">>> Start AEL stage1")
        algorithmEvolution = ael.AEL(use_local_llm, url,
            api_endpoint,api_key,llm_model,args.pop_size,args.n_pop,
            operators,args.m,operator_weights,load_data,output_path,debug_mode,evaluation=eva)

        # run AEL
        algorithmEvolution.run(object=False)
        print("AEL successfully finished !")
        get_output(eva.stages1, args.caption_path, args.image_path)
        if os.path.exists('results1.json'):
            os.remove('results1.json')
        if os.path.exists('objective1.json'):
            os.remove('objective1.json')
        with open('results1.json', 'w') as f1:
            json.dump(eva.stages1, f1)
        with open('objective1.json', 'w') as f2:
            json.dump(eva.objective1, f2)
    elif args.stage == 2:
        pipe = XFluxPipeline(args.model_type, device, args.offload)
        pipe.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
        with open('results1.json', 'r') as f1:
            results1 = json.load(f1)
        with open('objective1.json', 'r') as f2:
            objective1 = json.load(f2)
        min_key = min(objective1, key=lambda k: objective1[k])
        image_prompt = Image.open(results1[min_key])
        eva = Evaluation(pipe, image_prompt=image_prompt, stage=2, pattern=args.pattern, VLMpath=args.VLMpath)
        print(">>> Start AEL stage2")
        algorithmEvolution = ael.AEL(use_local_llm, url,
                                     api_endpoint, api_key, llm_model, args.pop_size, args.n_pop,
                                     operators, args.m, operator_weights, load_data, output_path, debug_mode,
                                     evaluation=eva)

        # run AEL
        algorithmEvolution.run(object=True)
        print("AEL successfully finished !")
        get_output(eva.stages2, args.caption_path, args.image_path)
        if os.path.exists('results2.json'):
            os.remove('results2.json')
        if os.path.exists('objective2.json'):
            os.remove('objective2.json')
        with open('results2.json', 'w') as f1:
            json.dump(eva.stages2, f1)
        with open('objective2.json', 'w') as f2:
            json.dump(eva.objective2, f2)




