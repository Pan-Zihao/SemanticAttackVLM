import collections
import json
import logging
import multiprocessing as mp
import os
import torch
from capture_metric.capture import CAPTURE
import nltk

# 下载 NLTK punkt 数据包
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# 设置日志
logging.basicConfig(level=logging.INFO)
eval_logger = logging.getLogger("lmms-eval")

# 这里定义你的参考和预测数据
refs = {
    'example_0': [
        "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. ",
        "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. "
    ],
}
preds = {
    'example_0': [
        "The image shows a red car, a white truck and other automobiles running on a city road. Pedestrians are walking on the side. Tall buildings can be seen under a clear blue sky."
    ]
}

def main():
    # 确保正确使用多进程
    if __name__ == '__main__':
        # 设置多进程启动方法为 fork
        torch.multiprocessing.set_start_method("spawn")
        #mp.set_start_method('fork', force=True)

        # 创建 CAPTURE 实例
        evaluator = CAPTURE()

        # 计算分数
        score = evaluator.compute_score(refs, preds)
        
        # 输出结果
        print(f"CAPTURE score: {score}")

# 调用 main 函数
if __name__ == '__main__':
    main()
