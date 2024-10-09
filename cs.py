import argparse
import os
import openai
import json
import torch
from PIL import Image
from diffusers import FluxPipeline
from x_flux.src.flux.xflux_pipeline import XFluxPipeline
from evaluation.VLMevaluation import caption_score, VQA_score
import shutil
import requests
from tqdm import trange

print(111)

