import time
import torch
import json
from PIL import Image
from evaluation.VLMevaluation import caption_score, VQA_score

class Evaluation:
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

