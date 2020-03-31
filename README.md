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

# 55 gtv maastro without generator, weighted_dice_coefficient_loss

# 56 gtv maastro with crop/rotate, weighted_dice_coefficient_loss

# 57 gtv maastro with crop/rotate/blur, weighted_dice_coefficient_loss

---

#60-89 different loss functions at 3 lrs

---

# testst for LR schedule

# 103 - focal-tversky-loss-0.0005

# 105 - focal-tversky-loss-lr (bad) (0.5, 5)

# 106 - focal-tversky-loss-lr-tight (0.8, 20)

# 107 - focal-tversky-loss-lr-tight-2 (0.8, 12)

# 108 - focal-tversky-loss-lr-tight-2-affine (0.8, 12) adds affine trans

# 109 - focal-tversky-loss-0.0005-augment (like 103 with affine and elastic, and more epochs) - multi(average) and single

# 110 - like 109 but with train on harvard-rt only - multi(average) and single

# 111 - like 109 but with train on maastro only - multi(average) and single (only did single for now, need plotting)

# 112, take model from 111 and train all layes on harvard-rt (only did single for now)

---

## test

# 23 - 37~40

NOTEs:
compare 45 to 50
then 50 to 51
