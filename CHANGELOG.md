# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
