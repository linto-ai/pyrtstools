# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Tee: Multiply single source to multiple sinks
- Tap: Open or close flow
- FileSink
- FileSource
- WebsocketSource
- AppSource
- API doc.

## [0.2.9] -2020-03-10
### Added
- PyRTSTools: Added runtime version check. pyrtstools.__version__
- KWS: Adds support for tensorflow Lite Compressed FlatBuffer format (.tflite) and keras (.h5/.hdf5) formats.

### Changed
- KWS: Model input shape is now fetched from the model. KWS input_shape parameter is now depreciated, it has been kept for backward compatibility but is ignored.
- KWS: Added debug parameter. If true, prediction values are prompted at each inference.
- VAD: Change default head and tail value to higher values.

## [0.2.8b] - 2020-02-13
### Added
- Adds CHANGELOG.md
- Adds manifest.json for version tracking.

### Changed
- Update setup.py for version control and add proper version dependency for tensorflow.
- Update README.md. 

### Removed
- Revomed speechpy mfcc, as it is redundant.

## [0.2.8a] - 2019-10-28
### Changed
- Changed the way .pb are loaded using serving dir instead of file path. Doesn't change API.
