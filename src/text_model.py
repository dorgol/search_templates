from transformers import CLIPProcessor, CLIPModel

# Instantiate model and processor
text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")