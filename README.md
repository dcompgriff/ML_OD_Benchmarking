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
nvidia-docker build -t gpu .
nvidia-docker run -it \
    --mount type=bind,source="$(pwd)",target=/ML_OD_Benchmarking \
    gpu /bin/bash





