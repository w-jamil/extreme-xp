#!/bin/bash
# Setup script to create data directory structure for each experiment

echo "Setting up data directory structure..."

# Create data directories for each experiment
mkdir -p batch/cyber
mkdir -p online/cyber
mkdir -p cl_case1/cyber
mkdir -p cl_case2/cyber

echo "Created directories:"
echo "  - batch/cyber/      (batch learning data)"
echo "  - online/cyber/     (online learning data)"
echo "  - cl_case1/cyber/   (continual learning case 1 data)"
echo "  - cl_case2/cyber/   (continual learning case 2 data)"
echo ""
echo "✅ Setup complete! Each experiment has its own data directory:"
echo "   - No conflicts between experiments"
echo "   - Automatic data download on first run"
echo "   - Independent caching for faster subsequent runs"
echo ""
echo "You can now run any experiment - they'll handle data automatically!"up script to create shared data directory structure

echo "Setting up shared data directory structure..."

# Create shared data directory
mkdir -p cyber

echo "Created directory:"
echo "  - cyber/     (shared data directory for all experiments)"
echo ""
echo "✅ Setup complete! All experiments now share the same data directory:"
echo "   - Download once → use everywhere"
echo "   - No data duplication" 
echo "   - Automatic data download on first run"
echo ""
echo "You can now run any experiment - they'll all use the same data!"
