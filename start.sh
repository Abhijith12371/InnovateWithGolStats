#!/bin/bash
echo "Installing wkhtmltopdf..."

# Install dependencies
nix-env -i wget

# Download static binary
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.centos7.x86_64.rpm
rpm2cpio wkhtmltox-0.12.6-1.centos7.x86_64.rpm | cpio -idmv

# Move to /usr/local/bin
mkdir -p /usr/local/bin
mv usr/local/bin/wkhtmltopdf /usr/local/bin/
mv usr/local/bin/wkhtmltoimage /usr/local/bin/

# Verify installation
if [ -f "/usr/local/bin/wkhtmltopdf" ]; then
    echo "wkhtmltopdf installed successfully at /usr/local/bin/wkhtmltopdf"
    
    # Start the application
    python exportVisualize.py
else
    echo "Installation failed!" >&2
    exit 1
fi

python exportVisualize.py