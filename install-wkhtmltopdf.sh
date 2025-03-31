#!/bin/bash
echo "Installing wkhtmltopdf..."

# Download the static binary (no root required)
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox-0.12.6.1-2.jammy-amd64.tar.xz
tar -xvf wkhtmltox-0.12.6.1-2.jammy-amd64.tar.xz
mv wkhtmltox/bin/wkhtmltopdf /usr/local/bin/

# Verify installation
if [ -f "/usr/local/bin/wkhtmltopdf" ]; then
    echo "wkhtmltopdf installed successfully at /usr/local/bin/wkhtmltopdf"
else
    echo "Installation failed!" >&2
    exit 1
fi