#!/bin/bash

# Create directories for better project structure
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/models
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/data
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/utils
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/scripts
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/outputs
mkdir -p /Users/dsroczyk/repos/Deep-Learning/transformers/notebooks

# Move model-related files
mv /Users/dsroczyk/repos/Deep-Learning/transformers/online_ast.py /Users/dsroczyk/repos/Deep-Learning/transformers/models/
mv /Users/dsroczyk/repos/Deep-Learning/transformers/online_ast_13.py /Users/dsroczyk/repos/Deep-Learning/transformers/models/
mv /Users/dsroczyk/repos/Deep-Learning/transformers/online_ast_11_sigmoid.py /Users/dsroczyk/repos/Deep-Learning/transformers/models/
mv /Users/dsroczyk/repos/Deep-Learning/transformers/ast_11_sigmoid_batch_padding.py /Users/dsroczyk/repos/Deep-Learning/transformers/models/

# Move data-related files
mv /Users/dsroczyk/repos/Deep-Learning/transformers/data/speech_commands_ds.py /Users/dsroczyk/repos/Deep-Learning/transformers/data/
mv /Users/dsroczyk/repos/Deep-Learning/transformers/data/data_preprocessing.py /Users/dsroczyk/repos/Deep-Learning/transformers/data/
mv /Users/dsroczyk/repos/Deep-Learning/transformers/data/audio_preprocessor.py /Users/dsroczyk/repos/Deep-Learning/transformers/data/

# Move utility files
mv /Users/dsroczyk/repos/Deep-Learning/transformers/nets/ensemble.py /Users/dsroczyk/repos/Deep-Learning/transformers/utils/

# Move notebook files
mv /Users/dsroczyk/repos/Deep-Learning/transformers/*.ipynb /Users/dsroczyk/repos/Deep-Learning/transformers/notebooks/

# Create a README file in each directory
echo "This directory contains model-related files." > /Users/dsroczyk/repos/Deep-Learning/transformers/models/README.md
echo "This directory contains data-related files and preprocessing scripts." > /Users/dsroczyk/repos/Deep-Learning/transformers/data/README.md
echo "This directory contains utility scripts and helper functions." > /Users/dsroczyk/repos/Deep-Learning/transformers/utils/README.md
echo "This directory contains scripts for running experiments or training." > /Users/dsroczyk/repos/Deep-Learning/transformers/scripts/README.md
echo "This directory contains output files such as logs, models, and results." > /Users/dsroczyk/repos/Deep-Learning/transformers/outputs/README.md
echo "This directory contains Jupyter notebooks for experiments and analysis." > /Users/dsroczyk/repos/Deep-Learning/transformers/notebooks/README.md

echo "✅ Project structure setup complete."
