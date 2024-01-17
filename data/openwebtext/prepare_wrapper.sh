# Environment setup
source /home/mdellag/miniconda3/etc/profile.d/conda.sh  # Initialize Conda
conda deactivate                                       # Deactivate any existing environment
conda activate nanoGPTenv                              # Activate the desired Conda environment

# Run the script, redirecting all output to a log file. 2>&1 redirects stderr to stdout.
python /home/mdellag/SplitNanoGPTdata/openwebtext/prepare.py > output.log 2>&1
