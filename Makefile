install:
	pip install --upgrade pip && pip install -e .
install_dev:
	pip install --upgrade pip && pip install -e .[dev]
install_requirements:
	pip install --upgrade pip && pip install -r requirements.txt
install_requirements_dev:
	pip install --upgrade pip && pip install -r requirements.txt -r dev-requirements.txt
lint:
	pylint --disable=R,C,W nalcos
all:
	install_dev lint
