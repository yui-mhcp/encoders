# :yum: CHANGELOG :yum:

This file tracks the major updates of the [encoders](https://github.com/yui-mhcp/encoders) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the embedding models, built on top of the `BaseModel` foundation (see the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) and the shared `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .` and pick a keras backend extra (`tf` or `torch`), plus the modality extras you need (`text`, `audio`, `image`, `datasets`, `dev`)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The minimum supported Python is now `3.12` (aligned with the keras backend wheels), and the version is single-sourced in `__version__`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured : the old `models/interfaces/base_*_model.py` classes are gone. Encoders now build on the `models/core` foundation (`base_model.py` composed from the `models/core/mixins/` — `audio`, `text`, `classification`, `processing`, `checkpoint`, `training`, ...) shared with `base_dl_project`
- **[BREAKING CHANGE]** the encoder classes now live in `models/encoder/` : `base_encoder.py` (abstract `BaseEncoder`, trained with the GE2E loss), `audio_encoder.py` (`AudioEncoder`) and `text_encoder.py` (`TextEncoder`, using pretrained embedding models)

### Minor updates

- New `models/encoder/rag.py` : a Retrieval-Augmented-Generation layer on top of `TextEncoder` (document ingestion / chunking, a `VectorDatabase` keyed by chunk hash, and semantic search)
- `TextEncoder` now defaults to the `BAAI/bge-m3` multilingual embedding model (`architectures/transformers/xlm_roberta_arch.py`)
- `custom_train_objects` cleanup : GE2E loss (`losses/ge2e_loss.py`) and metric (`metrics/equal_error_rate.py`), the `ge2e_generator` / `file_cache_generator` generators, `lr_schedulers`, and the `CheckpointManager` / `History` classes
- The `architectures` module keeps the generic layers, transformer architectures (`bart`, `bert`, `gpt2`, `mistral`, `t5`, `whisper`, `xlm_roberta`, ...) and the `simple_models` helpers
- The `information_retrieval.ipynb` notebook has been reworked around the new `TextEncoder` + RAG interface

### Known limitations

- The `AudioEncoder` / speaker-verification flow (`speaker_verification.ipynb`) is still being finalized as part of the migration from the legacy [siamese_networks](https://github.com/yui-mhcp/siamese_networks) project, and will be refined in a future update
