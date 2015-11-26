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
	# if already in bitbucket
	git co -b dev.. origin/dev..
	(to delete local branch) git branch -d name..
	(to delete remote branch) git push origin :name..

from time to time (while master keep changing)

	git fetch origin
	git rebase origin/master
	git rebase origin/dev.. (if needed)

when done

	git push -u origin dev.. (pull request) 

finally

	git rebase -i origin/master
	git push --force
	git co master
	(git pull)
	git merge --no-ff dev..
	git push
	
now the history is shorter and focused in master.


IDEAs
-----

hotpixels:
1 threshold first identified hotpixels
2 substitute with median average of 4 neighboring pixels
3 recursively identify new one until matching stop criteria
plt.contour(im[2])

### remember:
    mapmem for very large data (image5d)

feature.match_template(G[2], G[3])

feature.register_translation(G[0], G[1])
