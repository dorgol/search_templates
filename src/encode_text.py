class TextEmbeddings:
    def __init__(self, text, model, tokenizer):
        self.text = text
        self.model = model
        self.tokenizer = tokenizer

    def get_embeddings(self):
        inputs = self.tokenizer(text=self.text, padding=True, return_tensors="pt")
        outputs = self.model.get_text_features(**inputs)
        return outputs.detach()
