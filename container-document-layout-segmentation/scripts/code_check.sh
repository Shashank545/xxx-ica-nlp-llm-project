#!/usr/bin/env bash

set -e

pip3 install black \
             flake8 \
             bandit \
             safety

echo "[+]Running code style check by 'black'"
black document_layout_segmentation_service --check --line-length 100

echo "[+]Runing code style and code complexity check by 'flake8'"
flake8 --max-line-length 100 --extend-ignore=E203,W503 --extend-exclude='venv/' document_layout_segmentation_service

echo "[+]Running code security check by 'bandit'"
bandit -x tests -r document_layout_segmentation_service

echo "[+]Running dependency check by 'safety'"
safety check -r microservices/requirements.txt --full-report
