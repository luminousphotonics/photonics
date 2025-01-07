#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Build the React app
cd frontend
npm install
npm run build
cd ..

# Collect static files
python manage.py collectstatic --noinput