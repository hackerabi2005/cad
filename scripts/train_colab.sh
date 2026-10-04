#!/usr/bin/env bash
# Colab training script using google-colab-cli
set -e

SESSION_NAME="cad-train"

cleanup() {
    echo "Stopping Colab session $SESSION_NAME..."
    colab stop -s "$SESSION_NAME" || true
}
trap cleanup EXIT

echo "Starting Colab session..."
colab new -s "$SESSION_NAME"

echo "Installing requirements..."
colab install -s "$SESSION_NAME" -r ml/requirements.txt

echo "Packaging project ML assets..."
tar -czf bundle_input.tar.gz ml/ data/raw/

echo "Uploading project package to Colab..."
colab upload -s "$SESSION_NAME" bundle_input.tar.gz

echo "Running training launcher on Colab..."
colab exec -s "$SESSION_NAME" -f scripts/colab_launcher.py --timeout 1200

echo "Downloading generated artifacts from Colab..."
colab download -s "$SESSION_NAME" artifacts.tar.gz
tar -xzf artifacts.tar.gz -C ml/

echo "Colab training complete and artifacts synchronized!"
