#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Create static directory if it doesn't exist
mkdir -p static

# Create staticfiles directory if it doesn't exist
mkdir -p staticfiles

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate 