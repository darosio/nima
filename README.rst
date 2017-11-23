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
http://blogs.atlassian.com/2014/01/simple-git-workflow-simple/

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

	(git rebase origin/master)
	git rebase -i origin/master  (only before merging!!!)
	git push --force
		(git reset --hard origin/dev) in other repos
	git co master
	(git pull)
	git merge --no-ff dev..
	git push
	
now the history is shorter and focused in master.

in all local repos:
    git branch -d dev

TODO
----
* d_show(color=, colormap=False, im_print=True)
cm=plt.cm.Greens, cm=plt.cm.Reds, cm=plt.cm.Blues

* nimg buffer ZIP DARK FLAT -> float value, fig(image, profile, distribution)

# conclusion: Andor bg is  not flat and is changing in the timelapse
# TODO simulation of ratio on curved bg and subtraction of average
# best would be:
# flat-dark correct
# subtract Black (ideally with cells non transfected for AF correction)
# remember:: proper subtraction is especially important when weak and good signals mix.

To be used
----------

feature.match_template(G[2], G[3])

skimage.feature.register_translation(G[0], G[1])

### remember:
    mapmem for very large data (image5d)

IDEAs
-----

hotpixels:
1 threshold first identified hotpixels
2 substitute with median average of 4 neighboring pixels
3 recursively identify new one until matching stop criteria
plt.contour(im[2])

Doc
---

0. read
1. hotpixel
2. shading (clip=True)
3. bg (clip=True)

dict of 3D arrays, a 3D (time, x, y) array for each channel

It can be convenient to build a class around this data structure.

Tests
-----

skimage deprecate threshold_adaptive (for threshold_local) function.
tests are all passed but

> python test.py
output deprecation warning and RuntimeWarning at nimg.py:603
