import nltk
from nltk.translate.bleu_score import sentence_bleu

# 参考句子和预测句子
reference = "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. ".split()
candidate = "The image shows a red car, a white truck and other automobiles running on a city road. Pedestrians are walking on the side. Tall buildings can be seen under a clear blue sky.".split()



# 计算 BLEU 分数
bleu_score = sentence_bleu([reference], candidate)

print(f"BLEU score: {bleu_score}")
