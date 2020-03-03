#!/bin/bash

# figure our remote ssh
# install nvidia drivers
# install docker
# install docker nvidia
# mount external drives
# install screen


## script variables
path_to_key="-i /home/hosny/Dropbox/DFCI/00_access/R2-X2/R2-X2"
port="-p 1025"
user_ip="hosny@172.24.189.56"

ssh_connect="ssh -t -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $path_to_key $port $user_ip"
nvidia_watch="watch -n 0.5 nvidia-smi"

screen_docker="screen -S docker-screen sudo docker run -it --gpus all -v /home/hosny//segmentation/files:/files -v /home/hosny/segmentation/output:/output -v /mnt:/mnt dkr-model /bin/bash"

local_folder="/home/hosny/Dropbox/DFCI/30_prospective_validation/04_model/3d-unetplusplus"

# ## terminal 1 - monitor gpus
terminal_1="$ssh_connect '$nvidia_watch;command;/bin/bash'"
gnome-terminal -e "$terminal_1"

## terminal 2 - ssh, screen and docker
terminal_2="$ssh_connect '$screen_docker;command;/bin/bash'"
gnome-terminal -e "$terminal_2"

## terminal 3 - send code update to instance
gnome-terminal -e "$PWD/type_command.sh \"rsync --progress --stats -rave 'ssh $path_to_key $port -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' $local_folder/files/ $user_ip:/home/hosny/segmentation/files/\""

## terminal 4 - get output from AWs output folder
gnome-terminal -e "$PWD/type_command.sh \"rsync --progress --stats -rave 'ssh $path_to_key $port -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' $user_ip:/home/hosny/segmentation/output/ $local_folder/output/\""

# run locally
docker run -it -v $local_folder/files:/files -v $local_folder/output:/output -v /mnt:/mnt -p 8889:8888 dkr-model jupyter notebook --allow-root --ip=0.0.0.0
