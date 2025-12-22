# Changelog
This file keeps track of all notable changes between the different versions of cpe_search.

## v0.1.7 - 2025-12-22
### Added
- Improve CPE creation with subversion parts

## v0.1.6 - 2025-12-06
### Fixed
- Fix bugs from last update

## v0.1.5 - 2025-12-06
### Fixed
- Remove duplicate CPEs in results
- Remove created CPEs where a patch/subversion was contained in the CPE twice

## v0.1.4 - 2025-11-27
### Fixed
- Fixed bug with `-` and `_` in queries preventing valid CPE matches

## v0.1.3 - 2025-11-21
### Fixed
- Skip retrieval of deprecatedBy CPEs if NVD's dictionary does not contain this data

## v0.1.2 - 2025-11-18
### Fixed
- GitHub workflow to publish PyPI package uses more recent action versions

## v0.1.1 - 2025-11-18
### Added
- GitHub workflow to automatically publish a package to PyPI on new release

## v0.1.0 - 2025-11-17
### Added
- Initial release
