# 3d-unetplusplus

docker build -f dkr-model -t dkr-model .
docker run -it --gpus all -v\$PWD/files:/files -v /mnt:/mnt dkr-model /bin/bash

docker run -it --gpus all -v\$PWD/files:/files -v /mnt:/mnt -p 8886:8888 dkr-model /bin/bash

# to go into running container

docker exec -it <container name> /bin/bash

# notebook

jupyter notebook --allow-root --ip=0.0.0.0

## train

# 20-24 2d multi model

# 32 2d single model (lung first)

# 34 2d single model (lung middle, same order as planned)

only lung has ones

# 36 2d single model (no lung, removed from middle)

only cord has ones

---

# 37 3d single model for heart - dice

# 38 tversky loss - heart

# 39 tversky focal loss - heart

# 40 tversky focal loss 1/gamma - heart

---

# 41 dice ctv

# 42 tversky ctv ---x

# 43 ctv tversky focal

# 44 ctv tversky focal loss 1/gamma

---

# 45 ctv -tversky ---x

# 46 heart - tversky

# 47 lung - tversky

# 48 esophagus - tversky

# 49 cord - tversky

---

# 50 gtv- tversky

# 51 gtv - dice

# 52 gtv - tversky focal

# 53 gtv - tversky focal loss 1/gamma

# 54 gtv - bbox distance loss

---

---

## test

# 23 - 37~40

NOTEs:
compare 45 to 50
then 50 to 51
