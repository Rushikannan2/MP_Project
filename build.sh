#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Create static directory if it doesn't exist
mkdir -p static
mkdir -p staticfiles

# Download favicon files
curl -o static/favicon.ico https://raw.githubusercontent.com/Rushikannan2/MP_Project/main/static/favicon.ico
curl -o static/favicon-32x32.png https://raw.githubusercontent.com/Rushikannan2/MP_Project/main/static/favicon-32x32.png
curl -o static/favicon-16x16.png https://raw.githubusercontent.com/Rushikannan2/MP_Project/main/static/favicon-16x16.png

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate 