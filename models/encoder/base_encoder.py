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
import numpy as np

from loggers import timer
from utils import pad_to_multiple, pad_batch
from utils.callbacks import FunctionCallback, QueueCallback
from utils.keras import TensorSpec, ops
from ..core import BaseModel

logger = logging.getLogger(__name__)

class BaseEncoder(BaseModel):
    """
        Base class for Encoder architectures

        An encoder maps its inputs to a fixed-size vector space (the `embedding` space), in
        which similar inputs are close to each other according to `distance_metric`.

        `embed` is the canonical inference API (batched). `infer` embeds a single sample to
        comply with the `BaseModel` contract, making the generic `predict` usable as well.

        The evaluation of such models is mainly based on clustering evaluation : all inputs are embedded in the embedding space, and then clustered to group the similar embeddings into clusters. These clusters can be evaluated according to the expected (labels).
    """

    def __init__(self, distance_metric = 'cosine', ** kwargs):
        """
            Constructor for the base encoder configuration

            Arguments :
                - distance_metric   : the distance (or similarity) metric to use for the comparison of embeddings
        """
        self.distance_metric    = distance_metric

        super().__init__(** kwargs)

    def prepare_for_xla(self, *, inputs, pad_multiple = 128, ** kwargs):
        if self.pad_value is not None and ops.is_array(inputs) and ops.rank(inputs) in (2, 3):
            inputs = pad_to_multiple(
                inputs, pad_multiple, axis = 1, constant_values = self.pad_value
            )
        kwargs['inputs'] = inputs
        return kwargs

    @property
    def pad_value(self):
        return None

    @property
    def embedding_dim(self):
        return self.model.embedding_dim if hasattr(self.model, 'embedding_dim') else self.model.output_shape[-1]

    @property
    def output_signature(self):
        return TensorSpec(shape = (None, ), dtype = 'int32')

    @property
    def default_loss_config(self):
        return {'distance_metric' : self.distance_metric}

    @property
    def default_metrics_config(self):
        return self.default_loss_config

    def __str__(self):
        des = super().__str__()
        des += "- Embedding dim   : {}\n".format(self.embedding_dim)
        des += "- Distance metric : {}\n".format(self.distance_metric)
        return des

    def _maybe_init_loss_weights(self):
        if hasattr(self.loss, 'init_variables'):
            self.loss.init_variables(self.model)
            if not hasattr(self, 'process_batch_output'):
                self.process_batch_output = _reshape_labels_for_ge2e
                self._init_processing_functions()

    def compile(self, * args, ** kwargs):
        super().compile(* args, ** kwargs)

        self._maybe_init_loss_weights()

    def get_dataset_config(self, mode, ** kwargs):
        if self.pad_value is not None:
            kwargs.setdefault('pad_kwargs', {}).update({'padding_values' : (self.pad_value, 0)})

        return super().get_dataset_config(mode, ** kwargs)

    def prepare_dataset(self, dataset, mode, ** kwargs):
        from utils import is_dataframe
        from utils.datasets import prepare_dataset
        from custom_train_objects.generators import GE2EGenerator

        config = self.get_dataset_config(mode, ** kwargs)

        if is_dataframe(dataset):
            load_fn = config.pop('prepare_fn', None)
            if load_fn is not None and not hasattr(self, 'prepare_output'):
                load_fn     = self.prepare_input
                signature   = self.unbatched_input_signature
            else:
                signature   = [self.unbatched_input_signature, self.unbatched_output_signature]

            dataset = GE2EGenerator(
                dataset,
                load_fn = load_fn,
                output_signature    = signature,

                ** kwargs
            )
            logger.info('{} generator created :\n{}'.format(mode.capitalize(), dataset))
        elif isinstance(dataset, GE2EGenerator):
            config.pop('prepare_fn', None)
            if not dataset.batch_size:
                dataset.set_batch_size(config['batch_size'])
            else:
                config['batch_size'] = dataset.batch_size

        for k in ('cache', 'shuffle'): config[k] = False

        return prepare_dataset(dataset, ** config)

    @timer
    def embed(self,
              data,
              batch_size = 8,
              *,

              sort_by_length    = False,
              to_numpy  = True,
              tqdm  = lambda x: x,

              ** kwargs
             ):
        """
            Embed a (list of) data. This is the canonical inference API for encoders.

            Arguments :
                - data  : the data to embed, any type supported by `self.get_input`
                - batch_size    : the number of data to embed in parallel
                - sort_by_length    : whether to sort `data` by length before batching
                                      (reduces padding), the output order is preserved
                - to_numpy  : whether to convert the embeddings to `np.ndarray`
            Return :
                - embeddings    : the embedded data, a 2-D array `(len(data), embedding_dim)`
        """
        if hasattr(data, 'to_dict'):        data = data.to_dict('records')
        elif not isinstance(data, list):    data = [data]

        indices = None
        if sort_by_length and len(data) > batch_size:
            indices = sorted(range(len(data)), key = lambda idx: _get_sample_length(data[idx]))
            data    = [data[idx] for idx in indices]

        if self.runtime == 'keras': kwargs['as_dict'] = True
        elif self.runtime == 'hf':  kwargs['return_dict'] = False

        embeddings = []
        for s in tqdm(range(0, len(data), batch_size)):
            inputs = self.get_input(data[s : s + batch_size], ** kwargs)
            inputs = pad_batch(inputs, pad_value = self.pad_value)

            out = self.compiled_call(inputs, ** kwargs)
            if hasattr(out, 'output'): out = out.output
            if isinstance(out, dict):  out = out['dense']

            if to_numpy: out = ops.convert_to_numpy(out)
            embeddings.append(out)

        if len(embeddings) == 1:
            embeddings = embeddings[0]
        elif to_numpy:
            embeddings = np.concatenate(embeddings, axis = 0)
        elif self.runtime == 'keras':
            embeddings = ops.concatenate(embeddings, axis = 0)
        elif self.runtime in ('hf', 'trt', 'trt_llm', 'trt_llm_api'):
            import torch
            embeddings = torch.concat(embeddings, dim = 0)
        else:
            raise RuntimeError('Unspecified behavior for runtime {}'.format(self.runtime))

        if indices is not None:
            reverse_indices = np.argsort(indices)
            if isinstance(embeddings, np.ndarray) or self.runtime != 'keras':
                embeddings = embeddings[reverse_indices]
            else:
                embeddings = ops.gather(embeddings, reverse_indices, axis = 0)

        return embeddings

    def infer(self, data, *, callbacks = None, predicted = None, overwrite = False, return_output = True, ** kwargs):
        """
            Per-sample prediction (see the `BaseModel.infer` contract) : embeds a single
            sample by delegating to `embed`, the canonical (batched) API. Encoders have no
            persisted predictions, so the `predicted` / `overwrite` duplicate-detection
            mechanism does not apply.
        """
        output = {'embedding' : self.embed(data, ** kwargs)[0]}
        return self._finalize_predictions(
            output, callbacks = callbacks, predicted = predicted, return_output = return_output
        )

    def get_inference_callbacks(self, *, post_processing = None, ** kwargs):
        predicted, callbacks = {}, []

        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))

        return predicted, callbacks

    def filter_prediction_output(self, output):
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'distance_metric' : self.distance_metric})

        return config

def _get_sample_length(sample):
    try:    return len(sample)
    except TypeError: return 0

def _reshape_labels_for_ge2e(output, ** _):
    uniques, indexes = ops.unique(ops.reshape(output, [-1]), return_inverse = True)
    return ops.reshape(indexes, [ops.shape(uniques)[0], -1])
