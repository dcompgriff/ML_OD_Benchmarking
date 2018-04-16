README

Contents: Description of how to install and run facebookresearch's Detectron on AWS.
Website Resources:
1) Website for installing nvidia-docker on AWS: https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2
2) Website for

*
A) Need to install docker and docker-machine locally.
B) Need to have an AWS 
*

1) 




Cost record:
*3/30/18 1 Hr charge for EC2 Micro Instance ($0.006)
*3/30/18 1 Hr charge for EC2 p2.xlarge GPU instance ($1)



Command to create AWS instance:
docker-machine create --driver amazonec2 \
                      --amazonec2-region us-east-2 \
                      --amazonec2-zone a \
                      --amazonec2-ami ami-916f59f4 \
                      --amazonec2-instance-type p2.xlarge \
                      --amazonec2-vpc-id vpc-*** \
                      --amazonec2-access-key AKI********** \
                      --amazonec2-secret-key *************Fbf \
                      aws01



#####################################################################
#RUNNING THE CODE WITH DOCKER.
#####################################################################
0) Ssh into the gpu machine, and run "screen" to open a new screen session
(alternatively you can connect to a previous screen session). This is important
as screen allows you to reconnect to a previous ssh session and check the output
of a running job so that you don't have to stay ssh'd into the computer running
your desired job.
1) Run "./run_docker.sh". This command will build the desired docker container, and will
then run the docker container with the git repo (current directory) mounted to the
root folder at "/ML_OD_Benchmarking". It will also open up a bash terminal
at the root directory.
2) Run "cd /ML_OD_Benckmarking/detectron_scripts &", (Note this is run in the background
so you can track each log process)
3) Run "./benchmark.sh", which will run each of the deep nets using images
from "/ML_OD_Benchmarking/data/inputs/", and will output their results
to "/ML_OD_Benchmarking/data/outputs/<model_name>.json"





