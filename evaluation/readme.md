# LLaVA 安装指南

### 1. 克隆 LLaVA 仓库：
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

### 2. 设置 Conda 环境：
如果你还没有准备好的环境，可以创建一个包含 Python 3.10 的新环境，并安装必要的依赖：
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # 启用 PEP 660 支持
pip install -e .
```
**注意**：如果你已有可用的环境，可以跳过创建环境步骤，直接安装所需的包。

### 3. 拉取最新代码：
```bash
git pull
pip install -e .
```

### 4. 下载模型：
前往 [Hugging Face](https://huggingface.co/) 并下载以下模型到 evaluation/model 中：

- llava-v1.5-7b
- llava-v1.5-13b
- llava-v1.6-vicuna-7b
- llava-v1.6-vicuna-13b

详细的下载步骤可以参考[此指南](https://blog.csdn.net/qq_40600379/article/details/132006217)。

### 5. 模型调用：
你可以在 evaluation/VLMevaluation.py 中调整 model_path 从而正确的使用模型。但是需要注意的是，你需要把LLAVA/llava/eval/run_llava.py中第128行左右的print(outputs)改为return outputs
