#!/bin/bash  
# SUMO Installation Script - Fresh Installation  
# This script installs SUMO and all required dependencies from scratch  
  
set -e  # Exit on any error  
  
echo "🚀 Installing SUMO (Simulation of Urban MObility) - Fresh Installation"  
  
# Update system packages  
echo "📦 Updating system packages..."  
sudo apt-get update  
  
# Install system dependencies  
echo "🔧 Installing system dependencies..."  
sudo apt-get install -y software-properties-common wget curl git build-essential cmake python3 python3-pip python3-dev python3-venv libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev libgeos-dev libeigen3-dev swig default-jdk maven  
  
# Add SUMO PPA repository for latest version  
echo "📋 Adding SUMO official repository..."  
sudo add-apt-repository -y ppa:sumo/stable  
sudo apt-get update  
  
# Install SUMO  
echo "🏗️ Installing SUMO..."  
sudo apt-get install -y sumo sumo-tools sumo-doc  
  
# Set up environment variables  
echo "🌍 Setting up environment variables..."  
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc  
echo 'export PATH=$PATH:$SUMO_HOME/bin' >> ~/.bashrc  
echo 'export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools' >> ~/.bashrc  
  
# Apply environment variables to current session  
export SUMO_HOME=/usr/share/sumo  
export PATH=$PATH:$SUMO_HOME/bin  
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools  
  
# Create and activate virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies in virtual environment
echo "🐍 Installing Python dependencies in virtual environment..."
pip install --upgrade pip

# Install common SUMO Python packages
pip install traci sumolib matplotlib numpy pandas scipy

# Install project-specific requirements if they exist
if [ -f requirements.txt ]; then
    echo "📋 Installing project requirements..."
    pip install -r requirements.txt
fi
  
# Verify installation  
echo "✅ Verifying SUMO installation..."  
sumo --version  
  
# Test Python integration
echo "🧪 Testing Python integration..."
source venv/bin/activate
python -c "
import sys
try:
    import traci
    import sumolib
    print('✅ SUMO Python integration working correctly')
    print(f'   TraCI version: {traci.__version__ if hasattr(traci, \"__version__\") else \"Available\"}')
    print(f'   SUMO_HOME: {sumolib.checkBinary(\"sumo\")}')
except ImportError as e:
    print(f'❌ Python integration failed: {e}')
    sys.exit(1)
"
  
echo "🎉 SUMO installation completed successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. Restart your terminal or run: source ~/.bashrc"
echo "   2. Activate virtual environment: source venv/bin/activate"
echo "   3. Test with: sumo --version"
echo "   4. Run your SUMO simulations!"
