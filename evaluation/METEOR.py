import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 下载必要的数据
nltk.download('punkt')
nltk.download('wordnet')

# 创建词根还原器
lemmatizer = WordNetLemmatizer()

# 参考句子和生成句子
references = ["The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. "]
hypothesis ="The image shows a red car, a white truck and other automobiles running on a city road. Pedestrians are walking on the side. Tall buildings can be seen under a clear blue sky."


# 分词并进行词根还原
reference_tokens = [word_tokenize(ref) for ref in references]
hypothesis_tokens = word_tokenize(hypothesis)

# 词根还原处理
reference_tokens = [[lemmatizer.lemmatize(word) for word in ref] for ref in reference_tokens]
hypothesis_tokens = [lemmatizer.lemmatize(word) for word in hypothesis_tokens]

# 计算 METEOR 分数
score = meteor_score.single_meteor_score(reference_tokens[0], hypothesis_tokens)
print(f"METEOR score: {score:.4f}")
