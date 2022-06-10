# Changelog


## Version 0.4.0 - 2022-06-13
### Added
- [doc] Sphinx_click.

### Changed
- Click for the cli.
- Separated `nimg` from `bias dark|flat`.

### Removed
- docopt.
- [build] flake8-import-order. Will use isort in pre-commit.

### Fixed
- [build] Remove mypy cache every run.

## Version 0.3.5 - 2022-06-10
### Added
- [Build] mypy checking (yet imperfect).

### Changed
- Some plot graphics have improved.

## Version 0.3.4 - 2022-06-05
### Added
- Markdown for sphinx.

### Changed
- Changelog and authors from rst to md.

### Fixed
- matplotlib version for python-3.10.

## Version 0.3.3
-   Move out of flake8-based linting bandit; use system wide as:
```bandit -r src tests```
-   Move out safety; when updating packages dependencies consider:
```
poetry run safety check
poetry show --outdated
```
-   Update all packages except matplotlib (I will fix its tests).
-   Dropped python-3.7.
-   Added python-3.10.
-   Changed noxfile to use nox_poetry.

## Version 0.3.1
-   Dropping Pyscaffold in favor of poetry.
-   Works with python \< 3.7.

## Version 0.3
-   Transition to Pyscaffold template for faster dev cycles.

## Version 0.2.3
-   Works for clophensor data.
-   Heavy on memory.
-   Flat and Dark not tested.
