SHELL := /bin/bash

.DEFAULT_GOAL := help

VENV := .env
PYTHON := python3.8

.PHONY: virtualenv
virtualenv: ## Create virtualenv
	@if [ -d ${VENV} ]; then rm -rf ${VENV}; fi
	@mkdir ${VENV}
	${PYTHON} -m venv ${VENV}
	${VENV}/bin/pip install --upgrade pip==22.1.1
	${VENV}/bin/pip install -r requirements.txt
	${VENV}/bin/pip install pytest
	${VENV}/bin/pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
	#${VENV}/bin/pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

.PHONY: update-requirements-txt
update-requirements-txt: VENV := /tmp/venv/
update-requirements-txt: ## Update requirements.txt
	@if [ -d ${VENV} ]; then rm -rf ${VENV}; fi
	@mkdir ${VENV}
	${PYTHON} -m venv ${VENV}
	${VENV}/bin/pip install --upgrade pip==22.1.1
	${VENV}/bin/pip install -r unpinned_requirements.txt
	echo "# Created automatically by make update-requirements-txt. Do not update manually!" > requirements.txt

.PHONY: clean
clean: ## Clean python cache
	find . -type d -name "__pycache__" -exec rm -rf {} \;