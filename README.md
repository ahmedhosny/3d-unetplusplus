# 3d-unetplusplus

docker build -f dkr-model -t dkr-model .
docker run -it --gpus all -v\$PWD/files:/files -v /mnt:/mnt dkr-model /bin/bash

docker run -it --gpus all -v\$PWD/files:/files -v /mnt:/mnt -p 8886:8888 dkr-model /bin/bash

# to go into running container

docker exec -it <container name> /bin/bash

# notebook

jupyter notebook --allow-root --ip=0.0.0.0

# 20-24 2d multi model

# 32 2d single model (lung first)

# 34 2d single model (lung middle, same order as planned)

# 36 2d single model (no lung, removed from middle)

#
