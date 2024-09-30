from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "/data/codes/ty/CAPTURE/capture/ckpt/llava-v1.6-vicuna-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "/data/codes/ty/CAPTURE/capture/ckpt/llava-v1.6-vicuna-7b"
prompt = "Please describe the image in the following format:  a (picture/photo/watercolor/sketch)  of <number> <color> <object> <appearance> in the style of <style>.  <They/It/He/She> <gesture> on the <background describe> in the <location> on a <weather> day, <action description>, <environment description>."
image_file = "/data/codes/ty/evaluation/鸟人.png"

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

eval_model(args)