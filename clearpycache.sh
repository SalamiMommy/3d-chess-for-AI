#!/bin/bash


find . -type d -name '__pycache__' -exec rm -rf {} +
echo "All __pycache__ directories have been deleted."
