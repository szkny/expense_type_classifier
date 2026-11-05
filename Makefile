.PHONY: build install

build: clean
	@echo "Building the project..."
	python -m build
	@echo

install: build
	@echo "Installing the package..."
	pip install .
	@echo

editable-install: build
	@echo "Editable-installing the package..."
	pip install -e .
	@echo

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build dist *.egg-info
	@echo
