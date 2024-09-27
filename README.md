# SemanticAttackVLM 开发文档

## 当前版本

*2024.9.27* 
第一阶段代码初步完成，需要在服务器上使用FLUX进行测试。

潘子豪正在进行第二阶段的开发，婧怡正在进行第一阶段的测试和debug，童宇正在进行多模态大模型Benchmark的部署。

## 使用说明

### 第一阶段

使用LLM进化算法自动生成CLIPScore最小的图像-文本对，即对抗语义内容。
#### 存储目录

- images_stage1：存储LLM进化算法生成的所有的caption对应的图片，使用FLUX生成。 
- seedfile：存储输入LLM进化算法的种子文件
- seedfile/seedimage：存储输入LLM进化算法的种子caption的对应的图片，使用FLUX生成
- seedfile/seedcaption.txt：存储输入LLM进化算法的种子caption，使用GPT-4o生成
- stage1_results/images：存储第一阶段进化算法输出的最终caption对应的图片，使用FLUX生成
- stage1_results/captions/caption.txt：存储第一阶段进化算法输出的最终caption
- *其他文件均为LLM进化算法运行时产生的结果，理论上已经全部被接口封装*

#### 核心代码

- aell文件：core Lib for LLM进化算法
- llm_server：本地LLM接口和服务，但是我们使用的是api，所以没有什么用
- caption.txt：**注意这不是结果文件**，这是LLM进化算法运行过程中输入输出获得score的中间桥梁。
- generate_seeds.py：调用GPT-4o的api生成初始输入LLM进化算法的种子文件，结果放在seedfile
    
      python generate_seeds.py --QA_number 10 --captionfilename ./seedfile/seedcaption.txt --output_file ael_seeds/seeds.json
      
参数说明：``QA_number``为询问GPT-4o的轮数，但是得到的caption个数不一定等于这个，因为GPT-4o经常会一次性给出很多caption。具体数量以处理后的caption.txt为主。

- prepareAEL.py：主要是计算每个caption的得分功能的实现
- runAEL.py：核心代码，在seedfile生成之后运行LLM进化算法。

        python runAEL.py --object False
参数说明：都是默认参数，不需要更改。唯一的就是object，object可以保证在生成过程中prompt中``<object>``不发生改变，这主要应用于第二阶段使用IPAdapter，第一阶段默认即可。


- 注意：将所有FLUX和CLIP的路径改为本地路径。由于写代码的时候本地用不了FLUX和CLIP，所以核心部分并没有进行debug，需要注意debug一下。另外，LLM已经测试过，可以正常使用。