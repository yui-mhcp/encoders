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
from utils.keras import TensorSpec, ops
from ..interfaces import BaseModel

logger = logging.getLogger(__name__)

DEPRECATED_CONFIG = ('threshold', 'embed_distance')

class BaseEncoder(BaseModel):
    """
        Base class for Encoder architecture 
        
        The concept of Encoder is to have a unique model that "encodes" the inputs in an embedding space (named the `embedding`), and compares them based on a distance (or similarity) metric.
        
        The evaluation of such models is mainly based on clustering evaluation : all inputs are embedded in the embedding space, and then clustered to group the similar embeddings into clusters. These clusters can be evaluated according to the expected (labels). 
    """
    
    _directories    = {
        ** BaseModel._directories, 'embeddings_dir' : '{root}/{self.name}/embeddings'
    }
    
    def __init__(self, distance_metric = 'cosine', ** kwargs):
        """
            Constructor for the base encoder configuration
            
            Arguments :
                - distance_metric   : the distance (or similarity) metric to use for the comparison of embeddings
        """
        for k in DEPRECATED_CONFIG: kwargs.pop(k, None)
        
        self.distance_metric    = distance_metric
        
        super().__init__(** kwargs)
        
        self.__embeddings = None
    
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
    def embeddings(self):
        if self.__embeddings is None: self.__embeddings = self.load_embeddings()
        return self.__embeddings

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
    def embed(self, data, batch_size = 8, tqdm = lambda x: x, ** kwargs):
        """
            Embed a (list of) data
            
            Arguments :
                - data  : the data to embed, any type supported by `self.get_input`
                - batch_size    : the number of data to embed in parallel
            Return :
                - embeddings    : the embedded data
        """
        if hasattr(data, 'to_dict'):        data = data.to_dict('records')
        elif not isinstance(data, list):    data = [data]
        
        if self.runtime == 'keras': kwargs['as_dict'] = True
        elif self.runtime == 'hf':  kwargs['return_dict'] = False
        
        embeddings = []
        for s in tqdm(range(0, len(data), batch_size)):
            inputs = self.get_input(data[s : s + batch_size], ** kwargs)
            inputs = pad_batch(inputs, pad_value = self.pad_value)
            
            out = self.compiled_call(inputs, ** kwargs)
            if hasattr(out, 'output'): out = out.output
            if isinstance(out, dict):  out = out['dense']
            
            embeddings.append(ops.convert_to_numpy(out))
        
        return np.concatenate(embeddings, axis = 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({'distance_metric' : self.distance_metric})
        
        return config

def _reshape_labels_for_ge2e(output, ** _):
    uniques, indexes = ops.unique(ops.reshape(output, [-1]), return_inverse = True)
    return ops.reshape(indexes, [ops.shape(uniques)[0], -1])

