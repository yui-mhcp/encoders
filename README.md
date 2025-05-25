# :yum: Embedding models

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers               : custom layer implementations
│   ├── transformers         : transformer architecture implementations
│   ├── common_blocks.py     : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py  : utilities for text and sequence generation
│   ├── hparams.py           : hyperparameter management
│   └── simple_models.py     : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing
├── loggers                  : logging utilities for tracking experiment progress
├── models                   : main directory for model classes
│   ├── interfaces           : directories for interface classes
│   ├── encoder
│   │   ├── audio_encoder.py    : audio encoder class (audio to audio comparison with GE2E loss)
│   │   ├── base_encoder.py     : abstract Encoder class (trained with the GE2E loss)
│   │   └── text_encoder.py     : text encoder that uses pretrained embedding models
│   └── weights_converter.py : utilities to convert weights between different models
├── tests                    : unit and integration tests for model validation
├── utils                    : utility functions for data processing and visualization
├── information_retrieval.ipynb : notebook illustrating information retrieval with `TextEncoder`
├── LICENCE                  : project license file
├── README.md                : this file
├── requirements.txt         : required packages
└── speaker_verification.ipynb  : notebook illustrating speaker verification with `AudioEncoder` (will be updated in a future update)
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

**Important Note** : this project is the keras 3 extension of the [siamese network](https://github.com/yui-mhcp/siamese_networks) project. All features are not available yet. Once the convertion will be completely finalized, the siamese networks project will be removed in favor of this one. 

## Available models

| Input types   | Dataset   | Architecture  | Embedding dim | Trainer   | Weights   |
| :-----------: | :-------: | :-----------: | :-----------: | :-------: | :-------: |
| mel-spectrogram   | [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets) | `AudioEncoder (CNN 1D + LSTM)`   | 256   | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link)  |

Models should be unzipped in the `pretrained_models/` directory !

## Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Optimize `KNN` in pure `keras 3`
- [ ] Convert the `siamese_networks` project :
    - [x] Implement the `BaseEncoder` class
    - [ ] Implement the `BaseSiamese` class
    - [ ] Implement the `BaseComparator` class
    - [ ] Implement the `SiameseGenerator` class
    - [ ] Update the README to provide more information about evaluation of encoders
- [x] Implement text embedding models
- [x] Implement a `VectorDatabase` for information retrieval
- [x] Implement a `faiss`-based `VectorIndex` for information retrieval

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```