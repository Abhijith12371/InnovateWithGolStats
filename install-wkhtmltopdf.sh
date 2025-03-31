#!/bin/bash
echo "Installing wkhtmltopdf..."

# Install dependencies
apt-get update
apt-get install -y xfonts-75dpi xfonts-base wget xz-utils

# Download and install static binary
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.jammy_amd64.deb
dpkg -i wkhtmltox_0.12.6.1-2.jammy_amd64.deb

# Verify installation
if [ -f "/usr/local/bin/wkhtmltopdf" ]; then
    echo "wkhtmltopdf installed successfully at /usr/local/bin/wkhtmltopdf"
else
    echo "Installation failed!" >&2
    exit 1
fi