# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loggers import timer
from .base_encoder import BaseEncoder
from ..core import TextModelMixin

class TextEncoder(TextModelMixin, BaseEncoder):
    """
        Text embedding model

        The model is a pure `text -> embedding` primitive : `embed` (inherited from
        `BaseEncoder`) is the main API. The RAG layer built on top of it (database
        construction, document ingestion, semantic search) lives in `models/encoder/rag.py`.
    """
    pad_value   = TextModelMixin.blank_token_idx
    input_signature = TextModelMixin.text_signature
    prepare_input   = TextModelMixin.prepare_text

    def __init__(self, lang = 'multi', *, pretrained = 'BAAI/bge-m3', ** kwargs):
        kwargs.setdefault('tokenizer', pretrained)
        kwargs.setdefault('pretrained_name', pretrained)

        self._init_text(lang, ** kwargs)

        super().__init__(pretrained = pretrained, ** kwargs)

        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    def build(self, model = None, pretrained = None, ** kwargs):
        if model is None:
            from architectures.transformers import get_pretrained_transformer

            model = kwargs if not pretrained else get_pretrained_transformer(pretrained, ** kwargs)

        super().build(model = model)

    def __str__(self):
        return super().__str__() + self._str_text()

    @timer
    def retrieve(self, queries, database, ** kwargs):
        """ Embeds `queries` and returns the most relevant entries in `database` (see `rag.search`) """
        from .rag import search
        return search(self, database, queries, ** kwargs)

    def get_config(self):
        return {** super().get_config(), ** self.get_config_text()}
