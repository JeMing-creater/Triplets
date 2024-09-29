# coding=utf-8
import torch
from PIL import Image
import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipVisionModel
from transformers import AutoModelWithLMHead, AutoTokenizer

def blip(img_path):
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # model_name = 'blip-image-captioning-base'
    # model_downloaded = transformers.from_pretrained(model_name)
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    raw_image = Image.open(img_path).convert('RGB')

    # conditional image captioning
    text = "Using forceps to clamp the intestine"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def main():
    img_path = "/root/.cache/huggingface/forget/datasets/CholecT45/data/VID01/000000.png"
    blip(img_path)

if __name__ == "__main__":
    main()
