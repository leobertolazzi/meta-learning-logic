#!/bin/bash

source activate syllogistic_llms

echo "====================================================="
echo "Generating Tables and Figures...."
python src/process_results.py

echo ""
echo "====================================================="
echo "Running Error Analysis...."
python src/error_analysis/error_analysis.py
