from transformers import AutoProcessor, BlipModel

text_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
