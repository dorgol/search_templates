from transformers import AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering, BlipModel


class VideoModel:
    def __init__(self, processor_source, model_source, cls):
        self.cls = cls
        self.processor_source = processor_source
        self.model_source = model_source

    def get_processor(self):
        processor = AutoProcessor.from_pretrained(self.processor_source)
        return processor

    def get_model(self):
        model_func = getattr(self.cls, 'from_pretrained')
        model = model_func(self.model_source)
        return model

    def get_processor_model(self):
        processor = self.get_processor()
        model = self.get_model()
        return processor, model


vm_caption_processor = "Salesforce/blip-image-captioning-base"
vm_caption_model = "Salesforce/blip-image-captioning-base"

vm_vqa_processor = "Salesforce/blip-vqa-base"
vm_vqa_model = "Salesforce/blip-vqa-base"


vm_blip = VideoModel(vm_caption_processor, vm_caption_model, BlipModel)
vm_caption = VideoModel(vm_caption_processor, vm_caption_model, BlipForConditionalGeneration)
vm_vqa = VideoModel(vm_vqa_processor, vm_vqa_model, BlipForQuestionAnswering)

processor_blip, model_blip = vm_blip.get_processor_model()
processor_caption, model_caption = vm_caption.get_processor_model()
processor_vqa, model_vqa = vm_vqa.get_processor_model()

model_to_use = "openai/clip-vit-base-patch32"

