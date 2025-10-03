import torch
from transformers import AutoTokenizer, AutoModel
from .AbstractDenseModel import AbstractDenseModel


class AbstractTransformer(AbstractDenseModel):

    def __init__(self, *args, model_hgf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.model = AutoModel.from_pretrained(model_hgf)
        self.tokenizer = AutoTokenizer.from_pretrained(model_hgf)
        self.embeddings_dim = self.model.config.hidden_size

    def _tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_attention_mask=False, return_token_type_ids=False, add_special_tokens=True, return_tensors="pt")

    def encode_documents(self, texts):
        inputs = self._tokenize(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :] # Use the CLS token

    def encode_queries(self, texts):
        return self.encode_documents(texts)

    def start_multi_process_pool(self):
        return self.model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.model
