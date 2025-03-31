#!/bin/bash
echo "Installing wkhtmltopdf..."

# Create directory for wkhtmltopdf
mkdir -p /tmp/wkhtmltopdf

# Download the static binary from a reliable source
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.jammy_amd64.deb -P /tmp/wkhtmltopdf

# Extract the deb file manually (no dpkg needed)
cd /tmp/wkhtmltopdf
ar x wkhtmltox_0.12.6-1.jammy_amd64.deb
tar -xvf data.tar.xz

# Move the binary to /usr/local/bin
mv usr/local/bin/wkhtmltopdf /usr/local/bin/
mv usr/local/bin/wkhtmltoimage /usr/local/bin/

# Verify installation
if [ -f "/usr/local/bin/wkhtmltopdf" ]; then
    echo "wkhtmltopdf installed successfully at /usr/local/bin/wkhtmltopdf"
else
    echo "Installation failed!" >&2
    exit 1
fi