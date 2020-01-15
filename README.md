# 3d-unet-plus-plus

# mount aertslab volume

sudo mount -t cifs //rc-stor4.dfci.harvard.edu/aertslab /mnt/aertslab/ -o username=ah187,dir_mode=0777,file_mode=0777,nounix,vers=1.0

# transfer files to R2D2

rsync -rave "ssh -i R2-D2 -p 16594" /home/hosny/Dropbox/DFCI/30_prospective_validation/02_watershed ahmed@172.24.189.145:/home/ahmed/ahmed_watershed --delete

#

transfer files to R2Q5

rsync -rave "ssh -i R2-Q5 -p 16594" /home/hosny/Dropbox/DFCI/30_prospective_validation/02_watershed gpux@172.24.189.221:/home/gpux/ahmed_watershed --delete

# get files from R2D2

rsync -rave "ssh -i R2-D2 -p 16594" ahmed@172.24.189.145:/home/ahmed/ahmed_watershed /home/hosny/Dropbox/DFCI/30_prospective_validation/02_watershed/R2D2

# get files from R2Q5

rsync -rave "ssh -i R2-Q5 -p 16594" gpux@172.24.189.221:/home/gpux/ahmed_watershed /home/hosny/Dropbox/DFCI/30_prospective_validation/02_watershed/R2Q5

# docker

docker build -f dkr-model -t dkr-model .
docker run -it -v $PWD/data:/data -v$PWD/files:/files -v /mnt:/mnt dkr-model /bin/bash (without port mapping when not using jupyter)

docker run -it -v $PWD/data:/data -v$PWD/files:/files -v /mnt:/mnt -p 8886:8888 dkr-model /bin/bash

# to go into running container

docker exec -it <container name> /bin/bash

# notebook

jupyter notebook --allow-root --ip=0.0.0.0
