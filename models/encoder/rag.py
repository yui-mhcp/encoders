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

import logging
import hashlib

from loggers import timer
from utils import expand_path
from utils.databases import VectorDatabase
from utils.embeddings.index import NumpyIndex
from utils.text import parse_document, chunks_from_paragraphs, format_text

logger = logging.getLogger(__name__)

def get_text_hash(text):
    """ Return the deduplication key for an embedded text (see `build_database` `primary_key`) """
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def build_database(path = None, *, encoder = None, index = None, ** kwargs):
    """
        Creates (or reloads) a `VectorDatabase` designed for RAG

        Arguments :
            - path  : the database directory (`None` creates an in-memory database)
            - encoder   : the encoder the database is designed for, used to infer the
                          default `index` backend from its runtime
            - index : the `VectorIndex` backend to use
            - kwargs    : forwarded to `VectorDatabase`
        Return :
            - database  : a `VectorDatabase` with the canonical chunk `hash` as primary key
    """
    if index is None:
        if encoder is None:
            index = 'numpy'
        else:
            index = 'keras' if encoder.runtime == 'keras' else 'torch'

    return VectorDatabase(path, 'hash', index = index, ** kwargs)

@timer
def prepare_texts(texts, *, format = None, chunk_size = None, chunk_overlap = 0.2, ** kwargs):
    """
        Converts (a list of) raw text(s) to canonical chunks

        Arguments :
            - texts : a (list of) text(s), either `str`, either `dict` with a `text` entry
                      (and optional metadata, e.g. `source` / `filename` / `url`)
            - format    : a f-string or `jinja` template used to build the embedded text
            - chunk_size    : the maximum number of tokens per chunk
            - chunk_overlap : the (relative) number of tokens that overlap between chunks
            - kwargs    : forwarded to `chunks_from_paragraphs` (e.g., `group_by`, `tokenizer`)
        Return :
            - chunks    : a list of `dict` with (at least) the `text`, `source`, `formatted`
                          and `hash` entries
    """
    if hasattr(texts, 'to_dict'):       texts = texts.to_dict('records')
    elif not isinstance(texts, list):   texts = [texts]

    paragraphs = []
    for text in texts:
        if isinstance(text, str):   text = {'text' : text}
        else:                       text = text.copy()

        if 'source' not in text:
            for key in ('filename', 'url', 'conv_id'):
                if key in text:
                    text['source'] = text[key]
                    break
            else:
                text['source'] = 'raw'

        paragraphs.append(text)

    paragraphs = chunks_from_paragraphs(
        paragraphs, chunk_size, max_overlap_len = chunk_overlap, ** kwargs
    )

    return _finalize_chunks(paragraphs, format)

@timer
def prepare_documents(documents, *, format = None, chunk_size = None, chunk_overlap = 0.2, ** kwargs):
    """
        Parses then chunks (a list of) document(s) to canonical chunks

        Arguments :
            - documents : a (list of) document filename(s), forwarded to `parse_document`
            - format / chunk_size / chunk_overlap / kwargs  : see `prepare_texts`
        Return :
            - chunks    : see `prepare_texts` (`source` is the document filename)
    """
    documents = expand_path(documents)
    if not isinstance(documents, (list, tuple)): documents = [documents]

    inputs = []
    for doc in documents:
        if not isinstance(doc, str):
            raise ValueError('Unsupported document format : {}'.format(doc))

        chunks = parse_document(doc, ** kwargs)
        if chunk_size:
            chunks = chunks_from_paragraphs(
                chunks, chunk_size, max_overlap_len = chunk_overlap, ** kwargs
            )

        # multimodal chunks (image / audio / ...) are not embedded on their own : they are
        # attached to the `content` of the next text chunk
        mm_content = []
        for chunk in chunks:
            if chunk.get('type', 'text') == 'text':
                chunk['source'] = doc
                if mm_content:
                    if chunk.get('content', None):
                        chunk['content'] = mm_content + chunk['content']
                    else:
                        chunk['content'] = mm_content + [
                            {'type' : 'text', 'text' : chunk['text']}
                        ]
                    mm_content = []

                inputs.append(chunk)
            else:
                mm_content.append(chunk)

    return _finalize_chunks(inputs, format)

def _finalize_chunks(chunks, format = None):
    """ Sets the `formatted` (the effectively embedded text) and `hash` entries of each chunk """
    for chunk in chunks:
        chunk['formatted'] = format_text(format, ** chunk) if format else chunk.get('text', '')
        chunk['hash']      = get_text_hash(chunk['formatted'])
    return chunks

@timer
def index_data(encoder,
               database = None,
               *,

               texts    = None,
               documents    = None,

               group_by     = 'section',
               chunk_size   = 256,
               chunk_overlap    = 0.2,

               overwrite    = False,
               save     = None,

               ** kwargs
              ):
    """
        Embeds then adds the provided `texts` / `documents` to `database`

        This is the single write path : each source type is converted to canonical chunks
        (see `prepare_texts` / `prepare_documents`), which are then deduplicated (by their
        `hash` primary key), embedded (`encoder.embed`) and inserted.

        Arguments :
            - encoder   : the encoder model used to embed the chunks
            - database  : a `VectorDatabase`, or a path (`str`) to create/reload one, or
                          `None` to create an in-memory database
            - texts / documents : the data to index (see `prepare_*`)
            - group_by / chunk_size / chunk_overlap : chunking configuration
            - overwrite : whether to re-embed data already in the database
            - save      : whether to save the database (default : `True` if it has a `path`)
            - kwargs    : forwarded to the `prepare_*` functions and to `encoder.embed`
        Return :
            - database  : the (possibly created) `VectorDatabase`
    """
    assert texts is not None or documents is not None

    if not isinstance(database, VectorDatabase):
        database = build_database(database, encoder = encoder)

    if getattr(encoder, 'tokenizer', None) is not None:
        kwargs.setdefault('tokenizer', encoder.tokenizer)
    kwargs.update({
        'group_by' : group_by, 'chunk_size' : chunk_size, 'chunk_overlap' : chunk_overlap
    })

    chunks = []
    if texts:
        chunks.extend(prepare_texts(texts, ** kwargs))

    if documents:
        documents = expand_path(documents)
        if not isinstance(documents, (list, tuple)): documents = [documents]
        if not overwrite:
            indexed_sources = set(database.get_column('source'))

            kept = []
            for doc in documents:
                if doc in indexed_sources:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('The document {} is already in the database'.format(doc))
                else:
                    kept.append(doc)
            documents = kept

        if documents:
            chunks.extend(prepare_documents(documents, ** kwargs))

    # deduplication by primary key, both within the batch and against the database
    seen, unique = set(), []
    for chunk in chunks:
        if chunk['hash'] in seen: continue
        seen.add(chunk['hash'])

        if chunk['hash'] in database:
            if not overwrite: continue
            database.pop(chunk['hash'])
        unique.append(chunk)

    if unique:
        embeddings = encoder.embed(
            [chunk['formatted'] for chunk in unique],
            sort_by_length  = True,
            to_numpy    = isinstance(database.vectors, NumpyIndex),
            ** kwargs
        )
        database.multi_insert(unique, vectors = embeddings)

    if save is None: save = database.path is not None
    if save: database.save()

    return database

@timer
def search(encoder, database, queries, *, k = 10, sources = None, filters = None, reverse = False, ** kwargs):
    """
        Embeds `queries` then returns the most relevant entries in `database`

        Arguments :
            - encoder   : the encoder model used to embed the queries
            - database  : the `VectorDatabase` to search in
            - queries   : a (list of) text(s) to search
            - k     : the number of entries to retrieve per query
            - sources   : restricts the search to entries from the given source(s)
            - filters / reverse : see `VectorDatabase.search`
            - kwargs    : forwarded to `encoder.embed`
        Return :
            - results   : the retrieved entries, a list of `dict` (single query) or a list
                          of lists (list of queries), each `dict` having a `score` entry
    """
    if sources:
        if not isinstance(sources, (list, tuple, set)): sources = [sources]
        sources = set(sources)
        filters = {** (filters or {}), 'source' : lambda src: src in sources}

    embeddings = encoder.embed(
        queries, to_numpy = isinstance(database.vectors, NumpyIndex), ** kwargs
    )
    results = database.search(embeddings, k = k, filters = filters, reverse = reverse)

    return results[0] if not isinstance(queries, list) else results
