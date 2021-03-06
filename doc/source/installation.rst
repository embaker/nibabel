.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the NiBabel package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _installation:

************
Installation
************

NiBabel is a pure Python package at the moment, and it should be easy to get
NiBabel running on any system. For the most popular platforms and operating
systems there should be packages in the respective native packaging format
(DEB, RPM or installers). On other systems you can install NiBabel using
pip_ or by downloading the source package and running the usual ``python
setup.py install``.

.. This remark below is not yet true; comment to avoid confusion
   To run all of the tests, you may need some extra data packages - see
   :ref:`installing-data`.

Installer and packages
======================

.. _install_pypi:

pip and the Python package index
--------------------------------

If you are not using a Linux package manager, then best way to install NiBabel
is via pip_.  If you don't have pip already, follow the `pip install
instructions`_.

Then open a terminal (``Terminal.app`` on OSX, ``cmd`` or ``Powershell`` on
Windows), and type::

    pip install nibabel

This will download and install NiBabel.

If you really like doing stuff manually, you can install NiBabel by downoading
the source from `NiBabel pypi`_ .  Go to the pypi page and select the source
distribution you want.  Download the distribution, unpack it, and then, from
the unpacked directory, run::

    python setup.py install

or (if you need root permission to install on a unix system)::

    sudo python setup.py install

.. _install_debian:

Debian/Ubuntu
-------------

Our friends at NeuroDebian_ have packaged NiBabel at `NiBabel NeuroDebian`_.
Please follow the instructions on the NeuroDebian_ website on how to access
their repositories. Once this is done, installing NiBabel is::

  apt-get update
  apt-get install python-nibabel

Install from source
===================

If no installer or package is provided for your platform, you can install
NiBabel from source.

Requirements
------------

*  Python_ 2.6 or greater
*  NumPy_ 1.5 or greater
*  SciPy_ (for full SPM-ANALYZE support)
*  PyDICOM_ 0.9.7 or greater (for DICOM support)
*  `Python Imaging Library`_ (for PNG conversion in DICOMFS)
*  nose_ 0.11 or greater (to run the tests)
*  sphinx_ (to build the documentation)

Get the sources
---------------

The latest release is always available from `NiBabel pypi`_.

Alternatively, you can download a tarball of the latest development snapshot
(i.e. the current state of the *master* branch of the NiBabel source code
repository) from the `NiBabel github`_ page.

If you want to have access to the full NiBabel history and the latest
development code, do a full clone (aka checkout) of the NiBabel
repository::

  git clone git://github.com/nipy/nibabel.git

or::

  git clone https://github.com/nipy/nibabel.git

(The first may be faster, the second more likely to work behind a firewall).

Installation
------------

Just install the modules by invoking::

  sudo python setup.py install

If sudo is not configured (or even installed) you might have to use
``su`` instead.


Validating your install
-----------------------

For a basic test of your installation, fire up Python and try importing the module to see if everything is fine.
It should look something like this::

    Python 2.7.8 (v2.7.8:ee879c0ffa11, Jun 29 2014, 21:07:35)
    [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import nibabel
    >>>


To run the nibabel test suite, from the terminal run ``nosetests nibabel`` or ``python -c "import nibabel; nibabel.test()``.

To run an extended test suite that validates ``nibabel`` for long-running and resource-intensive cases, please see :ref:`advanced_testing`.

.. include:: links_names.txt
