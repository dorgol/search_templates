from transformers import CLIPProcessor, CLIPModel

# Instantiate model and processor
text_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
