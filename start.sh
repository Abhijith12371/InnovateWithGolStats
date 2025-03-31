#!/bin/bash
echo "Installing wkhtmltopdf..."

# Install dependencies
apt-get update
apt-get install -y wget xfonts-75dpi xfonts-base

# Download and install wkhtmltopdf
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb
dpkg -i wkhtmltox_0.12.6-1.bionic_amd64.deb
apt-get install -f -y  # Fix dependencies

# Verify installation
if [ -f "/usr/bin/wkhtmltopdf" ]; then
    echo "wkhtmltopdf installed successfully at /usr/bin/wkhtmltopdf"
else
    echo "Installation failed!" >&2
    exit 1
fi

# Start the application
python exportVisualize.py