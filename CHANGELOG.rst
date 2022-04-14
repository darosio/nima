=========
Changelog
=========


https://keepachangelog.com/en/1.0.0/

Version 0.3.3
=============
- Move out of flake8-based linting bandit; use system wide as::

	bandit -r src tests

- Move out safety; when updating packages dependencies consider::

	poetry run safety check
	poetry show --outdated

- Update all packages except matplotlib (I will fix its tests)
- Dropped python-3.7
- Added python-3.10
- Changed noxfile to use nox_poetry

Version 0.3.1
=============
- Dropping Pyscaffold in favor of poetry
- Works with python < 3.7

Version 0.3
===========
- Transition to Pyscaffold template for faster dev cycles.

Version 0.2.3
=============
- Works for clophensor data
- Heavy on memory
- Flat and Dark not tested.

Version 0.2
===========

Version 0.1
===========

- Feature A added
