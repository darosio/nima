===============================
nimg
===============================

.. image:: https://img.shields.io/travis/darosio/nimg.svg
        :target: https://travis-ci.org/darosio/nimg

.. image:: https://img.shields.io/pypi/v/nimg.svg
        :target: https://pypi.python.org/pypi/nimg


Image analysis scripts based on scipy.ndimage and skimage.

* Free software: New BSD license
* Documentation: https://nimg.readthedocs.org.

Features
--------

* TODO

Git flow
--------

	git co -b dev..

from time to time (while master keep changing)

	git fetch origin
	git rebase origin/master
	git rebase origin/dev.. (if needed)

when done

	git push -u origin dev.. (pull request) 

finally

	git rebase -i origin/master
	git push -force (but you may want to keep the branch as it is on remote \*)
	git co master
	(git pull)
	git merge --no-ff dev..
	git push   (\* now the history is shorter and focused in master)

