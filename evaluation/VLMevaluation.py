import random
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from rouge import Rouge

def calculate_rouge(reference_summary, generated_summary):
    rouge = Rouge()

    # 计算 ROUGE 分数
    scores = rouge.get_scores(generated_summary, reference_summary)[0]

    # 计算平均 F1 分数
    average_f1 = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    return average_f1


def image_caption(model_path, image_file, prompt):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    result = eval_model(args)
    return result

# 使用示例
#model_path = "model/llava-v1.5-7b"
#model_path = "model/llava-v1.5-13b"
#model_path = "model/llava-v1.6-vicuna-7b"
#model_path = "model/llava-v1.6-vicuna-13b"
#image_file = "鸟人.png"
#prompt = "Please describe the image in the following format: a (picture/photo/watercolor/sketch) of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture> on the <background describe> in the <location> on a <weather> day, <action description>, <environment description>."

#result = image_caption(model_path, image_file, prompt)
#print(result)


#TODO
def caption_score(caption,model_path, image_file, prompt):
    pre_caption = image_caption(model_path, image_file, prompt)
    #score = random.random()
    score = calculate_rouge(caption,pre_caption)
    return score


#TODO
def VQA_score(caption, image):
    score = random.random()
    return score
