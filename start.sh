#!/bin/bash
echo "Installing wkhtmltopdf..."

# Download and install wkhtmltopdf
apt-get update
apt-get install -y wget
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb
dpkg -i wkhtmltox_0.12.6-1.bionic_amd64.deb
apt-get install -f -y  # Fix dependencies

# Ensure wkhtmltopdf is available in the path
export PATH=$PATH:/usr/local/bin
echo "wkhtmltopdf installed successfully!"

# Start the application
python exportVisualize.py
