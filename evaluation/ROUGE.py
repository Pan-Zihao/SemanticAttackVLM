from rouge import Rouge

# 定义参考摘要和生成的摘要
reference_summary = "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. "
generated_summary = "The image shows a red car, a white truck and other automobiles running on a city road. Pedestrians are walking on the side. Tall buildings can be seen under a clear blue sky."

# 创建 ROUGE 对象
rouge = Rouge()

# 计算 ROUGE 分数
scores = rouge.get_scores(generated_summary, reference_summary)

# 打印结果
print("ROUGE Scores:")
for score in scores:
    print(score)
