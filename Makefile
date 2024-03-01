# run all python tests in this folder

# Makefile default shell is /bin/sh which does not implement source.

SHELL := /bin/bash
.ONESHELL: 
.PHONY:

build:
	python3 -m build

install:
	python3 -m pip install --upgrade --no-index --no-build-isolation dist/CBOSS*.whl

test:
	export PYTHONPATH=$(CURDIR)/..; python -m unittest discover . -v

requirements:
	python3 -m pipreqs.pipreqs . --force