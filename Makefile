# Makefile for Cardio3D AI: Setup, Training, Testing, and Serving

.PHONY: setup train-local train-colab test serve clean build-3d build-web

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
	pytest ml/tests/ -v
	pytest api/tests/ -v
	python scripts/test_ui_playwright.py

build-web:
	cd web && npm run build

serve:
	python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

report:
	python scripts/generate_pdf_report.py
