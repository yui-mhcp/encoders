# :yum: Encoder networks

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── custom_architectures
├── custom_layers
├── custom_train_objects
│   ├── generators
│   │   ├── file_cache_generator.py  : abstract generator caching processed data
│   │   └── ge2e_generator.py        : generator dedicated to format GE2E input
│   ├── losses
│   │   └── ge2e_loss.py    : Keras 3 implementation of the GE2E loss
│   ├── metrics
│   │   └── ge2e_metric.py  : custom metric associated to the GE2E loss
├── loggers
├── models
│   ├── encoder
│   │   ├── audio_encoder.py    : audio encoder class (audio to audio comparison with GE2E loss)
│   │   └── base_encoder.py     : abstract Encoder class (trained with the GE2E loss)
├── pretrained_models
├── unitests
├── utils
└── speaker_verification.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

**Important Note** : this project is the keras 3 extension of the [siamese network](https://github.com/yui-mhcp/siamese_networks) project. All features are not available yet. Once the convertion will be completely finished, the siamese networks project will be removed in favor of this one. 

## Available models

| Input types   | Dataset   | Architecture  | Embedding dim | Trainer   | Weights   |
| :-----------: | :-------: | :-----------: | :-----------: | :-------: | :-------: |
| mel-spectrogram   | [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets) | `AudioEncoder (CNN 1D + LSTM)`   | 256   | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link)  |

Models must be unzipped in the `pretrained_models/` directory !

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/encoders.git`
2. Go to the root of this repository : `cd encoders`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Optimize `KNN` in pure `keras 3`
- [x] Implement the `clustering` procedure
- [ ] Implement the `similarity matrix` evaluation procedure
- [ ] Implement the `clustering` evaluation procedure
- [ ] Convert the `siamese_networks` project :
    - [x] Implement the `BaseEncoder` class
    - [ ] Implement the `BaseSiamese` class
    - [ ] Implement the `BaseComparator` class
    - [ ] Implement the `SiameseGenerator` class
    - [ ] Update the README to provide more information about evaluation of encoders

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

Tutorials : 
- [Medium tutorial for speaker verification with siamese networks](https://medium.com/analytics-vidhya/building-a-speaker-identification-system-from-scratch-with-deep-learning-f4c4aa558a56). 
- [Google GE2E Loss tutorial](https://google.github.io/speaker-id/publications/GE2E/) : amazing Google tutorial explaining the benefits of the GE2E loss compared to the Siamese approach (which is really similar to their `Tuple End-to-End (TE2E) loss` principle)

Github project : 
- [voicemap project](https://github.com/oscarknagg/voicemap) : nice project for speaker verification.
- [OpenAI's CLIP](https://github.com/openai/clip) : the official `CLIP` implementation in pytorch. 
