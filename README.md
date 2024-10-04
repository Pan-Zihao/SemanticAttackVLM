# SemanticAttackVLM 开发文档

## 当前版本

*2024.9.27* 
第一阶段代码初步完成，需要在服务器上使用FLUX进行测试。

*2024.9.29* 
第一阶段代码（使用CLIPScore）跑通。

*2024.9.30*
VLM benchmark（caption）完成。

*2024.10.3*    
- 将CLIPScore替换为benchmark captionscore。     
- 将gpt-4o替换为claude-3.5。    
- 第二阶段代码初步完成，需要在服务器上调试。   

## 使用说明
首先使用`generate_seeds.py`生成种子，再使用`runAEL.py`进行LLM进化算法。
### generate_seeds.py

询问LLM的轮数，注意这不一定等于种子的数量，因为LLM一次response可能会给出很多答案。

    parser.add_argument('--QA_number',default=10, type=int)
种子caption的存储位置，注意区分是第一阶段还是第二阶段。
    
    parser.add_argument('--captionfilename',default = "./seedfile1/seedcaption.txt", type=str)
种子图像的存储位置，注意区分是第一阶段还是第二阶段。    

    parser.add_argument('--imagefilename',default = "./seedfile1/seedimage", type=str)
种子json文件的输出地址，这里第一阶段和第二阶段生成的文件名称是相同的，注意清除。    
    
    parser.add_argument('--output_file',default='ael_seeds/seeds.json')
为第一阶段生成种子还是第二阶段，两者的区别体现在函数initialize里面，以及prompt的不同。    
    
    parser.add_argument('--stage',default=1, type=int)
VLM模型路径，比如"model/llava-v1.5-7b"，根据服务器实际路径。

    parser.add_argument('--VLMpath', default = "model/llava-v1.5-7b", type=str)
评分模式，可以选择是使用caption打分，还是VQA打分。

    parser.add_argument('--pattern', default="caption", type=str)
一些ipadapter相关的参数，根据服务器实际路径。

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
基本流程：  
- stage1：初始化prompt $\rightarrow$ 实例化FLUX $\rightarrow$ 调用initialize函数（生成caption及对应图片，评分） $\rightarrow$ 结果写入seed.json
- stage2：根据results1.json选出分数最低的图文对作为种子 $\rightarrow$ 提取并保存object, 初始化prompt $\rightarrow$ 实例化x_flux_ipadapter $\rightarrow$ 调用initialize函数（生成caption及对应图片，评分） $\rightarrow$ 结果写入seed.json

### prepareAEL.py
主要是Evaluation类，这是在进化算法运行过程中为每一个样本提供分数的。接下来会讲解一下Evaluation类的设计：  
        
    def __init__(self, T2Imodel, image_prompt, stage, pattern, VLMpath):
            print("begin evaluate")
            self.stages1 = {}
            self.stages2 = {}
            self.objective1 = {}
            self.objective2 = {}
            self.index = 0
            self.T2I = T2Imodel
            self.image_prompt = image_prompt
            self.stage = stage
            self.pattern = pattern
            self.VLMpath = VLMpath
- `T2Imodel`是传入用于生成caption对应的图片的模型，stage1是FLUX，stage2是FLUX-ipadapter，因为需要传入stage1的结果作为image prompt。
- `image_prompt`是ipadapter需要的图片输入，这只在stage2有效，stage1设置为None。
- `stage`一个整数，决定处于哪个阶段。
- `pattern`是评测的模式，根据VLM生成的caption进行打分还是VQA打分。
- `VLMpath`是VLM模型（比如LLaVA）的路径，根据服务器实际路径。
- `stage1/2`是一个字典，存储所有caption以及对应的图片地址，key为caption，value为图片地址。
- `objective1/2`是一个字典，存储所有的caption以及对应的分数，key为caption，value为分数。

      def get_score(self, caption):
          print(caption)
          if self.stage == 1:
              image = self.T2I(
                  caption,
                  height=1024,
                  width=1024,
                  guidance_scale=3.5,
                  num_inference_steps=50,
                  max_sequence_length=512,
                  generator=torch.Generator("cpu").manual_seed(0)
              ).images[0]
              image_path = f"./images_stage1/{self.index}.png"
              image.save(image_path)
              self.stages1[caption] = image_path
第一阶段的评测过程，没什么好说的。

        elif self.stage == 2:
            image = self.T2I(
                prompt=caption,
                width=1024,
                height=1024,
                guidance=4,
                num_steps=25,
                seed=self.index,
                true_gs=3.5,
                timestep_to_start_cfg=5,
                image_prompt=self.image_prompt,
                ip_scale=1.0,
            )
            image_path = f"./images_stage2/{self.index}.png"
            image.save(image_path)
            self.stages2[caption] = image_path
        if self.pattern == 'caption':
            fitness = caption_score(caption, self.VLMpath, image_path)
        elif self.pattern == 'VQA':
            fitness = VQA_score(caption, self.VLMpath, image_path)
        if self.stage == 1:
            self.objective1[caption] = fitness
        elif self.stage == 2:
            self.objective2[caption] = fitness
        self.index = self.index + 1
        return fitness
先根据`stage`判断是哪个阶段的生成过程，然后选择评测模式，给出评测分数。`index`是图片的编号。

### runAEL.py

    # number of algorithms in each population, default = 10
    parser.add_argument('--pop_size', default=50, type=int)
    # number of populations, default = 10
    parser.add_argument('--n_pop',default=5,type=int)
    # number of parents for 'e1' and 'e2' operators, default = 2
    parser.add_argument('--m',default=4,type=int)
这些参数都是LLM进化算法的参数，`pop_size`决定了每一步选多少个最优的。

    parser.add_argument('--caption_path', default="./stage1_results/captions/captions.txt")
    parser.add_argument('--image_path', default="./stage1_results/images")
以上分别是最终结果的保存地址，注意区分是第一阶段还是第二阶段。

最终所有caption对应的图片地址，以及所有caption对应的分数这两组映射关系分别保存为两个字典，字典序列化为json文件，第一阶段就是
`results1.json`和`objective1.json`，第二阶段就是1改成2。