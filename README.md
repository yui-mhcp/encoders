# :yum: Embedding models

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.1.0-orange.svg)

This repository implements **encoder models**, i.e. models that map their inputs (audio, text, ...) to a fixed-size vector space (the *embedding* space) in which similar inputs are close to each other. It powers two main use-cases : **speaker verification** (audio) and **information retrieval / RAG** (text).

Check the [CHANGELOG](CHANGELOG.md) file for the latest updates of this project, and the [global CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) for a cross-project overview ! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   │   ├── *_arch.py                : concrete transformer models (bart, bert, gpt2, mistral, t5, whisper, xlm_roberta, ...)
│   │   ├── text_transformer_arch.py : base blocks for text-based Transformers
│   │   └── transformer_arch.py      : generic Transformer blocks
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   └── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing
│   ├── callbacks                : callbacks loading and implementations
│   ├── generators               : custom data generators
│   │   ├── file_cache_generator.py : generator caching processed samples to disk
│   │   └── ge2e_generator.py       : batch sampler for the GE2E loss (speakers × utterances)
│   ├── losses                   : loss functions
│   │   ├── ge2e_loss.py             : Generalized End-to-End (GE2E) loss
│   │   └── loss_with_multiple_outputs.py : utility class allowing losses to output additional metrics
│   ├── metrics                  : metrics loading and implementations
│   │   └── equal_error_rate.py      : Equal Error Rate (EER) metric for verification
│   ├── optimizers               : optimizer loading
│   │   └── lr_schedulers.py         : custom learning-rate schedulers
│   ├── checkpoint_manager.py    : handle model checkpoint management (inspired from `tf.train.CheckpointManager`)
│   └── history.py               : main History class to compute training statistics / track config
├── loggers                  : custom utilities for the `logging` module*
├── models                   : main directory for model classes
│   ├── core                     : the `BaseModel` foundation (mixins + saving / migration utils)*
│   ├── encoder                  : encoder model classes
│   │   ├── base_encoder.py          : abstract `BaseEncoder` (maps inputs to embeddings, trained with the GE2E loss)
│   │   ├── audio_encoder.py         : `AudioEncoder` (audio-to-audio comparison for speaker verification)
│   │   ├── text_encoder.py          : `TextEncoder` (text embeddings using pretrained models, e.g. `bge-m3`)
│   │   └── rag.py                   : Retrieval-Augmented-Generation layer (vector database + semantic search)
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models        : pretrained checkpoints (e.g., the `bge-m3` text encoder)
├── tests                    : `pytest` suite mirroring the `utils` / `loggers` and model tree*
├── utils                    : shared data-processing / keras utilities*
├── information_retrieval.ipynb : notebook illustrating information retrieval / RAG with `TextEncoder`
├── speaker_verification.ipynb  : notebook illustrating speaker verification with `AudioEncoder` (WIP)
├── CHANGELOG.md                : project-specific changelog
├── CITATION.cff                : citation metadata
├── CONTRIBUTING.md             : contribution guidelines
├── INSTALLATION.md             : shared GPU environment installation guide
├── LICENCE                     : project license file
├── pyproject.toml              : project metadata, dependencies and tooling config
├── README.md                   : this file
└── __version__.py              : single-source project version
```

\* The `loggers`, `utils`, `models/core` and most of the `tests` tree are the shared foundation maintained in the [data_processing](https://github.com/yui-mhcp/data_processing) and [base_dl_project](https://github.com/yui-mhcp/base_dl_project) repositories — check them for more information on these modules and the `BaseModel` interface.

**Important Note** : this project is the keras 3 extension of the [siamese networks](https://github.com/yui-mhcp/siamese_networks) project. All features are not available yet : the `TextEncoder` / RAG flow is finalized, while the `AudioEncoder` / speaker-verification flow is still being converted. Once the conversion is completely finalized, the siamese networks project will be removed in favor of this one.

## Available models

| Input types   | Dataset   | Architecture  | Embedding dim | Trainer   | Weights   |
| :-----------: | :-------: | :-----------: | :-----------: | :-------: | :-------: |
| text          | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `TextEncoder (XLM-RoBERTa)` | 1024  | [BAAI](https://huggingface.co/BAAI) | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| mel-spectrogram   | [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets) | `AudioEncoder (CNN 1D + LSTM)`   | 256   | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link)  |

Models should be unzipped in the `pretrained_models/` directory !

## Installation and usage

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/encoders.git`
2. Go to the root of this repository : `cd encoders`
3. Install the package with a backend and the modality you need : `pip install -e .[tf,text]` (or `pip install -e .[torch,text]`)
4. Open an example notebook and follow the instructions !

Like `base_dl_project`, this project trains / runs real models, so it needs a **keras backend**. Pick the one matching your setup — the extras pull `keras>=3` plus the backend :

```bash
pip install -e .[tf]         # keras + tensorflow
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# modality extras (pick the encoder you use) :
pip install -e .[text]       # TextEncoder + RAG (transformers, sentencepiece, document parsers)
pip install -e .[audio]      # AudioEncoder (librosa, scipy, ...)
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas)
pip install -e .[dev]        # test tooling (pytest & plugins)

# extras can be combined :
pip install -e .[tf,text,audio]
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the example notebooks.

### Quickstart

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models.encoder import TextEncoder

# 1. Load a pretrained text encoder (multilingual bge-m3) ...
model = TextEncoder(pretrained = 'BAAI/bge-m3', name = 'bge-m3')

# 2. ... and embed some texts into the vector space
embeddings = model.embed(['Hello world !', 'Bonjour le monde !'])   # -> (2, 1024) array

# 3. Build a RAG index over your texts / documents (the single write path) ...
from models.encoder.rag import index_data, search

database = index_data(model, texts = ['... your paragraphs ...'], documents = ['my_doc.pdf'])

# 4. ... then run a semantic search over it
query = 'What is an embedding ?'

res = search(model, database, query, k = 3)
```

**[EXPERIMENTAL] Accelerated inference** : the `TextEncoder` can also run on the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) `LLM API` (`runtime = 'trt_llm_api'`, `encode_only` mode), which packs variable-length batches without padding (×1.75–2.9 faster than `transformers` on mixed-length batches, measured in `float16` on an RTX 5090). **Known limitation** : `bge-m3` (`XLMRobertaModel`) is not yet supported by the upstream TensorRT-LLM library (tested with `1.3.0rc29`), so this runtime currently requires a TensorRT-LLM build with `XLMRoberta` support for the pytorch backend.

See the `information_retrieval.ipynb` notebook for the full walkthrough (document ingestion, vector database, semantic search). The `speaker_verification.ipynb` notebook illustrates the `AudioEncoder` flow (still being finalized). :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list :

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Optimize `KNN` in pure `keras 3`
- [x] Implement text embedding models
- [x] Implement a `VectorDatabase` for information retrieval
- [x] Implement a `VectorIndex` for information retrieval
- [x] Implement the RAG layer (`models/encoder/rag.py`) on top of `TextEncoder`
- [x] Support the `TensorRT-LLM` `LLM API` for text encoders (experimental)
- [ ] Convert the `siamese_networks` project :
    - [x] Implement the `BaseEncoder` class
    - [ ] Implement the `BaseSiamese` class
    - [ ] Implement the `BaseComparator` class
    - [ ] Implement the `SiameseGenerator` class
    - [ ] Update the README to provide more information about evaluation of encoders

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
