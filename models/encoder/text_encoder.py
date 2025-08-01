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

from loggers import timer
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
                
                ** kwargs
               ):
        assert texts is not None or documents
        
        if texts is None:                   texts = []
        elif hasattr(texts, 'to_dict'):     texts = texts.to_dict('records')
        elif not isinstance(texts, list):   texts = [texts]
        
        if database is None:
            if save:
                if directory is None: directory = self.pred_dir
                path = os.path.join(directory, path)

            database = init_database(
                'VectorDatabase',
                path = path,
                primary_key = ('source', 'formatted'),
                vector_key  = 'embedding',
                embedding_dim   = self.embedding_dim,
                ** kwargs
            )

        inputs = texts.copy()
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                inputs[i] = {'text' : inputs[i], 'source' : 'raw'}
            elif 'source' not in inputs[i]:
                if 'filename' in inputs[i]: inputs[i]['source'] = inputs[i]['filename']
                elif 'url' in inputs[i]:    inputs[i]['source'] = inputs[i]['url']
                else:                       inputs[i]['source'] = 'raw'

        if inputs:
            if chunk_size:
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
            documents = expand_path(documents)
            if not isinstance(documents, (list, tuple)): documents = [documents]
            
            if format:
                _format_chunk = lambda _format, ** kwargs: format_text(_format, ** kwargs)
            else:
                _format_chunk = lambda _format, text = None, ** _: text
            
            _db_files   = set(database.get_column('filename'))
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

                for i in reversed(range(len(chunks))):
                    if not chunks[i].get('text', None):
                        del chunks[i]
                    else:
                        chunks[i].update({
                            'source' : doc, 'formatted' : _format_chunk(format, ** chunks[i])
                        })
                inputs.extend(chunks)
        
        if inputs:
            sorted_indices  = sorted(
                range(len(inputs)), key = lambda idx: len(inputs[idx]['formatted'])
            )
            sorted_inputs = [inputs[idx] for idx in sorted_indices]
            
            embeddings = self.embed([inp['formatted'] for inp in inputs], ** kwargs)
            embeddings = embeddings[np.argsort(sorted_indices)]
            
            database.extend(inputs, vectors = embeddings)
            if save: database.save()
        
        return database

    def retrieve(self, queries, database, ** kwargs):
        return database.search(self.embed(queries), ** kwargs)
        
    def predict_old(self,
                texts,
                batch_size  = 8,
                *,
                
                primary_key = 'text',
                
                group_by    = 'filename',
                chunk_size  = 256,
                chunk_overlap   = 0.2,
                
                save    = True,
                vectors = None,
                filename    = 'embeddings.h5',
                directory   = None,
                overwrite   = False,
                
                ** kwargs
               ):
        if isinstance(texts, (str, dict)):  texts = [texts]
        elif hasattr(texts, 'to_dict'):     texts = texts.to_dict('records')
        
        paragraphs = []
        for text in texts:
            if not isinstance(text, str):
                paragraphs.append(text)
            elif '*' not in text and not is_path(text):
                paragraphs.append({'text' : text})
            else:
                try:
                    paragraphs.extend(parse_document(f, ** kwargs))
                except Exception as e:
                    logger.warning('An exception occured while parsing file {} : {}'.format(f, e))
                
        if chunk_size:
            paragraphs = chunks_from_paragraphs(
                paragraphs,
                chunk_size,
                group_by    = group_by,
                max_overlap_len = chunk_overlap,
                
                tokenizer   = self.text_encoder,
                ** kwargs
            )
        
        if not os.path.dirname(filename):
            if directory is None: directory = self.pred_dir

            filename = os.path.join(directory, filename)
        
        if vectors is None:
            vectors  = build_vectors_db(filename)
        
        queries = paragraphs
        if vectors is not None and not overwrite:
            paragraphs = [p for p in paragraphs if p not in vectors]
        
        if paragraphs:
            vectors = self.embed(
                paragraphs,
                batch_size  = batch_size,
                primary_key = primary_key,
                
                reorder = False,
                initial_results = vectors,
                
                ** kwargs
            )
            if save:
                vectors.save(filename, overwrite = True)
                vectors = vectors[queries]

        return vectors
    
    def get_config(self):
        return {** super().get_config(), ** self.get_config_text()}
    