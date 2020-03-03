#!/bin/bash

## user set variables
# the aws instance IP - user is ubuntu
aws_ip="ec2-3-81-96-55.compute-1.amazonaws.com"
# path to ssh key
path_to_key="/home/hosny/Dropbox/DFCI/00_access/aws-segmentation/aws-segmentation.pem"
# working within the instance
working_dir="cd /ahmed/code/3d-unetplusplus"
screen_docker="screen -S docker-screen docker run -it --gpus all -v /ahmed/code/files:/files -v /ahmed/output:/output -v /ahmed/data:/data dkr-model /bin/bash"
# working locally
local_code_folder="/home/hosny/Dropbox/DFCI/30_prospective_validation/04_model/3d-unetplusplus"

## script variables
ssh_connect="ssh -t -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i $path_to_key ubuntu@$aws_ip"
nvidia_watch="watch -n 0.5 nvidia-smi"

# ## terminal 1 - monitor aws gpus
terminal_1="$ssh_connect '$nvidia_watch;command;/bin/bash'"
gnome-terminal -e "$terminal_1"

## terminal 2 - ssh, screen and docker
terminal_2="$ssh_connect '$screen_docker;command;/bin/bash'"
gnome-terminal -e "$terminal_2"
#
## terminal 3 - send code update to AWS instance
gnome-terminal -e "$PWD/type_command.sh \"rsync --progress --stats --rsync-path='sudo rsync' -rave 'ssh -i $path_to_key -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' $local_code_folder/files ubuntu@$aws_ip:/ahmed/code\""

## terminal 4 - get output from AWs output folder
gnome-terminal -e "$PWD/type_command.sh \"rsync --progress --stats --rsync-path='sudo rsync' -rave 'ssh -i $path_to_key -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' ubuntu@$aws_ip:/ahmed/output $local_code_folder\""

# ## terminal 5 (same terminal) - open local jupyter notebook in local docker
# # in lab # --gpus all
docker run -it -v $local_code_folder/files:/files -v $local_code_folder/output:/output -v /mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA:/data -v /mnt:/mnt -p 8889:8888 dkr-model jupyter notebook --allow-root --ip=0.0.0.0

# not in lab (only get RTOG) # --gpus all
# docker run -it -v $local_code_folder/files:/files -v $local_code_folder/output:/output -v /home/hosny/Desktop/RTOG:/data -p 8889:8888 dkr-model jupyter notebook --allow-root --ip=0.0.0.0
