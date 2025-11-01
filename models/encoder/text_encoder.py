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

import os
import logging
import numpy as np

from loggers import Timer, timer
from utils import is_path, expand_path
from utils.text import parse_document, chunks_from_paragraphs, format_text
from utils.databases import init_database
from utils.keras import TensorSpec, ops
from .base_encoder import BaseEncoder
from ..interfaces.base_text_model import BaseTextModel

logger = logging.getLogger(__name__)

class TextEncoder(BaseTextModel, BaseEncoder):
    pad_value   = BaseTextModel.blank_token_idx
    input_signature = BaseTextModel.text_signature
    prepare_input   = BaseTextModel.prepare_text
    
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

    @timer(name = 'prediction')
    def predict(self,
                texts   = None,
                *,
                
                documents   = None,
                
                format  = None,
                
                group_by    = 'section',
                chunk_size  = 256,
                chunk_overlap   = 0.2,
                
                save    = True,
                path    = 'database.db',
                directory   = None,
                overwrite   = False,
                database    = None,
                index_type  = None,
                remove_unspecified_documents    = False,
                
                ** kwargs
               ):
        assert texts is not None or documents
        
        if texts is None:                   texts = []
        elif hasattr(texts, 'to_dict'):     texts = texts.to_dict('records')
        elif not isinstance(texts, list):   texts = [texts]
        
        if database is None:
            if save:
                if directory is None: directory = self.pred_dir
                os.makedirs(directory, exist_ok = True)
                path = os.path.join(directory, path)

            if not index_type:
                index_type = '{}Index'.format(
                    'keras' if self.runtime == 'keras' else 'torch'
                )
            
            database = init_database(
                'VectorDatabase',
                path = path,
                primary_key = ('source', 'formatted'),
                
                index   = index_type,
                vector_key  = 'embedding',
                embedding_dim   = self.embedding_dim,
                ** kwargs
            )

        inputs = texts.copy()
        if inputs:
            with Timer('inputs processing'):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], str):
                        inputs[i] = {'text' : inputs[i], 'source' : 'raw'}
                    elif 'source' not in inputs[i]:
                        if 'filename' in inputs[i]: inputs[i]['source'] = inputs[i]['filename']
                        elif 'url' in inputs[i]:    inputs[i]['source'] = inputs[i]['url']
                        else:                       inputs[i]['source'] = 'raw'

                inputs = chunks_from_paragraphs(
                    inputs,
                    chunk_size,
                    group_by    = group_by,
                    max_overlap_len = chunk_overlap,

                    tokenizer   = self.tokenizer,
                    ** kwargs
                )

                if format:
                    for inp in inputs:
                        inp['formatted'] = format_text(format, ** inp)
                else:
                    for inp in inputs: inp['formatted'] = inp['text']

                if not overwrite:
                    inputs = [inp for inp in inputs if inp not in database]
        
        if documents:
            with Timer('documents processing'):
                documents = expand_path(documents)
                if not isinstance(documents, (list, tuple)): documents = [documents]

                if format:
                    _format_chunk = lambda _format, ** kwargs: format_text(_format, ** kwargs)
                else:
                    _format_chunk = lambda _format, text = None, ** _: text

                _db_files   = set(database.get_column('source'))
                if remove_unspecified_documents:
                    to_remove = _db_files.difference(set(documents))
                    if to_remove: database.filter(source = lambda s: s in to_remove)
                
                for doc in documents:
                    if not isinstance(doc, str):
                        raise ValueError('Unsupported document format : {}'.format(doc))

                    if not overwrite and doc in _db_files:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('The document {} is already in the database'.format(doc))
                        continue

                    chunks = parse_document(doc, ** kwargs)
                    if chunk_size:
                        chunks = chunks_from_paragraphs(
                            chunks,
                            chunk_size,
                            group_by    = group_by,
                            max_overlap_len = chunk_overlap,

                            tokenizer   = self.tokenizer,
                            ** kwargs
                        )

                    _uniques = set()
                    for i in reversed(range(len(chunks))):
                        if not chunks[i].get('text', None):
                            del chunks[i]
                        else:
                            chunks[i].update({
                                'source' : doc, 'formatted' : _format_chunk(format, ** chunks[i])
                            })
                            if chunks[i]['formatted'] in _uniques:
                                del chunks[i]
                            else:
                                _uniques.add(chunks[i]['formatted'])
                        
                    inputs.extend(chunks)
        
        if inputs:
            sorted_indices  = sorted(
                range(len(inputs)), key = lambda idx: len(inputs[idx]['formatted'])
            )
            sorted_inputs = [inputs[idx] for idx in sorted_indices]
            
            embeddings = self.embed(
                [inp['formatted'] for inp in sorted_inputs], to_numpy = False, ** kwargs
            )
            if ops.is_tensor(embeddings):
                embeddings = ops.gather(embeddings, np.argsort(sorted_indices), axis = 0)
            else:
                embeddings = embeddings[np.argsort(sorted_indices)]
            
            database.extend(inputs, vectors = embeddings)
            if save: database.save()
        
        return database

    def retrieve(self, queries, database, ** kwargs):
        return database.search(self.embed(queries, to_numpy = False), ** kwargs)
    
    def get_config(self):
        return {** super().get_config(), ** self.get_config_text()}
    