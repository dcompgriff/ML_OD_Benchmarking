#!/bin/bash

# Run the demo CNN image labeling.
mkdir /tmp/test_out

python hello_condor_docker.py

# Zip the output files in output.zip
tar -cvf output.tar /tmp/test_out






