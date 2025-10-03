from .AbstractTransformer import AbstractTransformer


class Cocondenser(AbstractTransformer):

    def __init__(self, model_hgf="Luyu/co-condenser-marco-retriever"):
        super().__init__(model_hgf=model_hgf)

        self.name = "co-condenser"
        self.embeddings_dim = 768
