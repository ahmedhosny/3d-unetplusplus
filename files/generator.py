from scipy import ndimage
import numpy as np

def crop_helper(arr, z_shift, y_shift, x_shift, final_shape):
    return arr[z_shift:z_shift+final_shape[0], y_shift:y_shift+final_shape[1], x_shift:x_shift+final_shape[2]]

def crop_arr(image, label, final_shape):
    #
    z_diff = image.shape[0] - final_shape[0]
    y_diff = image.shape[1] - final_shape[1]
    x_diff = image.shape[2] - final_shape[2]
    #
    assert z_diff >= 0, "final shape cannot be bigger the input shape along the z axis"
    assert y_diff >= 0, "final shape cannot be bigger the input shape along the y axis"
    assert x_diff >= 0, "final shape cannot be bigger the input shape along the x axis"
    #
    z_shift = np.random.randint(0, z_diff+1)
    y_shift = np.random.randint(0, y_diff+1)
    x_shift = np.random.randint(0, x_diff+1)
    #
    image = crop_helper(image, z_shift, y_shift, x_shift, final_shape)
    label = crop_helper(label, z_shift, y_shift, x_shift, final_shape)
    return image, label

def rotate_helper(arr, z_angle, y_angle, x_angle):
    """
    Rotates along all 3 axis. Other angles not used due to computation time.
    """
    arr = ndimage.rotate(arr, float(z_angle), order=3, axes=(1,2), mode="constant", cval = 0, reshape=False)
    return arr

def rotate_arr(image, label, angle_range):
    """
    Other angles not used due to computation time.
    """
    start = angle_range*-1
    end = angle_range+1
    #
    z_angle = np.random.randint(start, end)
    y_angle = np.random.randint(start, end)
    x_angle = np.random.randint(start, end)
    #
    image = rotate_helper(image, z_angle, y_angle, x_angle)
    label = rotate_helper(label, z_angle, y_angle, x_angle)
    return image, label

def blur_arr(label, blur_multiplier, blur_random_range):
    """
    Sigma: standard deviation for gaussian blur - must be positive.
    The larger the sigma, the more powerful the applied blur.
    """
    ratio = (len(label[label==1]) / label.size) * 100
    sigma = ratio * blur_multiplier
    #
    start = sigma * blur_random_range * -1
    end = sigma * blur_random_range
    #
    random_shift = np.random.uniform(start, end)
    #
    sigma = sigma + random_shift
    label = ndimage.filters.gaussian_filter(label.astype(np.float32), sigma, mode='constant')
    return label

def pepper_arr(arr):
    """
    Adds Gausian noise with std of 25 HU, equivilant to 0.0061 because the images are already remaped.
    """
    noise = np.random.normal(0, 0.0061, arr.shape)
    return arr + noise

def augment_data(images, labels, batch_size, angle_range, final_shape, blur_multiplier, blur_random_range, augment=True):
    """
    Will randomize samples at each run.
    Will crop out samples and hence allows the generator to always return complete batches.
    """
    image_list = []
    label_list = []
    for index, (image, label) in enumerate(zip(images, labels)):
        assert image.shape == label.shape, "image and label do not have the same shape."
        # augment
        if augment:
            # image, label = rotate_arr(image, label, angle_range)
            image, label = crop_arr(image, label, final_shape)
            # noise
            # image = pepper_arr(image)
            # label = blur_arr(label, blur_multiplier, blur_random_range)
        # reshape
        image = image.reshape(1, *image.shape)
        label = label.reshape(1, *label.shape)
        # append
        image_list.append(image)
        label_list.append(label)
        # print("generator augmentation :: {}".format(index))
    #
    no_of_samples = len(image_list)
    # randomize
    idx = np.random.permutation(no_of_samples)
    # crop out leftover data to get complete batches
    data_len_for_complete_batches = no_of_samples - (no_of_samples%batch_size)
    no_of_batches = data_len_for_complete_batches / batch_size
    # format and return
    images = np.split( np.array(image_list,'float32')[idx][:data_len_for_complete_batches], no_of_batches)
    labels = np.split( np.array(label_list,'float32')[idx][:data_len_for_complete_batches], no_of_batches)
    return images, labels

def generator(images, labels, batch_size, angle_range, final_shape, blur_multiplier, blur_random_range, augment):
    while True:
        images_batches, labels_batches = augment_data(images,
            labels, batch_size, angle_range, final_shape,
            blur_multiplier, blur_random_range, augment)

        for image_batch, label_batch in zip(images_batches, labels_batches):
            yield image_batch, label_batch



 # if add_noise and np.random.randint(3) == 0:
 #        # Add uncorrelated gaussian pixel noise with random variance (1,200)
 #        fluc = np.random.randint(1,200)
 #        cur_img = np.random.normal(cur_img,fluc)
 #        cur_img *= (cur_img>0)
 #        cur_img = np.round(cur_img).astype( dtype=np.float32)
