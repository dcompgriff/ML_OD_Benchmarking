#!/usr/bin/env python

with open('/tmp/test_out/generated_output.txt', 'w+') as f:
	f.write("This is generated test output to test the docker output mechanism.\n")

print("Hello Condor Dockerfile Image!")
