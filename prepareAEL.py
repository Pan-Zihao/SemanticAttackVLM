import time
import torch
from PIL import Image

class Evaluation:
    def __init__(self, T2Imodel, Encoder, Processor, device):
        print("begin evaluate")
        self.index = 0
        self.T2I = T2Imodel
        self.encoder = Encoder
        self.processor = Processor
        self.device = device
    def evaluate(self):
        time.sleep(1)
        try:
            with open("caption.txt", "r", encoding="utf-8") as file:
                caption = file.read()
            score = self.get_score(caption)
            return score
        except Exception as e:
            print("Error:",str(e))
            return None

    def get_score(self, caption):
        print(caption)
        image = self.T2I(
            caption,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save(f"./images_stage1/{self.index}.png")
        image_v = Image.open(f"./images_stage1/{self.index}.png")
        inputs = self.processor(text=caption, images=image_v, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        outputs = self.encoder(**inputs)
        fitness = outputs.logits_per_image[0]
        self.index = self.index + 1

        return fitness


class GetPrompts:
    def __init__(self):
        self.prompt_task = "You need to change the caption in the equation."
        self.prompt_func_name = "captionUpdate"
        self.prompt_func_inputs = [""]
        self.prompt_func_outputs = ["caption"]
        self.prompt_inout_inf = "You only need to change the content of the caption in the method."

        self.prompt_other_inf = "return caption:The changed value"

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

