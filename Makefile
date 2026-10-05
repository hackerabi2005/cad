# Makefile for Cardio3D AI: Setup, Training, Testing, and Serving

.PHONY: setup train-local train-colab test test-ui serve clean build-3d build-web report

setup:
	pip install -r ml/requirements.txt
	cd web && npm install

build-3d:
	python scripts/build_heart_glb.py

train-local:
	python -m ml.train

train-colab:
	bash scripts/train_colab.sh

test:
	python -m pytest ml/tests/ api/tests/ -v

test-ui:
	python scripts/test_ui_playwright.py

build-web:
	cd web && npm run build

serve:
	python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"

report:
	python scripts/generate_pdf_report.py
