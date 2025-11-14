# Dockerfile for HERO-EVCS
# Provides a complete environment with SUMO and all dependencies
# Works on Windows, macOS, and Linux via Docker
# Optimized for running main.py on any platform
# Uses Ubuntu 24.04 to match paper experimental environments
FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies with retry logic for network issues
# The DNS configuration in /etc/resolv.conf gets reset by Docker's internal DNS,
# so we rely on Docker's host network configuration instead
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg \
    gpg-agent \
    dirmngr \
    wget \
    curl \
    git \
    ca-certificates \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libxerces-c-dev \
    libfox-1.6-dev \
    libgdal-dev \
    libproj-dev \
    libgl2ps-dev \
    libgeos-dev \
    libeigen3-dev \
    swig \
    default-jdk \
    maven \
    && rm -rf /var/lib/apt/lists/*

# Add SUMO PPA repository and install SUMO
RUN add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# Set SUMO environment variables
# Initialize PYTHONPATH properly to avoid undefined variable warning
ENV SUMO_HOME=/usr/share/sumo
ENV PATH=$PATH:$SUMO_HOME/bin
ENV PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$SUMO_HOME/tools

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Upgrade pip first, then install core packages, then requirements
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install core SUMO Python packages first
RUN pip3 install --no-cache-dir traci sumolib

# Install other essential packages
RUN pip3 install --no-cache-dir matplotlib numpy pandas scipy

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt || \
    (echo "Some optional packages failed to install, continuing..." && true)

# Copy project files (data/ folder should be mounted as volume)
COPY . /app

# Make main.py executable
RUN chmod +x /app/main.py

# Verify SUMO and OSMnx installation
RUN sumo --version && \
    python3 -c "import traci, sumolib; print('✅ SUMO Python integration OK')" && \
    python3 -c "import osmnx as ox; print(f'✅ OSMnx {ox.__version__} installed')" && \
    python3 -c "import sys; sys.path.insert(0, '/app'); print('✅ Project path configured')"

# Create entrypoint script for running main.py
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    echo "================================================"
    echo "  HERO-EVCS Docker Container"
    echo "================================================"
    echo ""
    echo "Usage:"
    echo "  docker run <image> python3 main.py --city \"City Name\" --total-stations 50"
    echo ""
    echo "Examples:"
    echo "  docker run <image> python3 main.py --city \"Singapore\" --total-stations 50"
    echo "  docker run <image> python3 main.py --city \"Mumbai, India\" --total-stations 30 --max-episodes 10"
    echo ""
    echo "To run bash interactively:"
    echo "  docker run -it <image> /bin/bash"
    echo ""
    exit 0
fi

# Execute the command
exec "$@"
EOF

RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
    
# Default command (can be overridden)
CMD ["python3", "main.py", "--help"]