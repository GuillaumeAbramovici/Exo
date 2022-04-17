import glob
import time
from PIL import Image
import cv2
import numpy as np
import math
import imageio
import matplotlib.pyplot as plt


def fill_holes(image):

    # Read image
    # im_in = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    # th, im_th = cv2.threshold(im_in, 150, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = image.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv


    # Display images.
    # cv2.imshow("Thresholded Image", image)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)
    return im_out

def segmentation(image, number_of_iterations):
    # read the image
    image = cv2.imread(image)
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.02)
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    # plt.imshow(segmented_image)
    # plt.show()
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 0
    masked_image[labels == cluster] = [0, 0, 0]
    masked_image[labels == 1] = [255, 255, 255]
    # convert back to original shape

    ## Create a mask to only keep the first
    masked_image = masked_image.reshape(image.shape)
    mask = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel=np.ones((number_of_iterations,number_of_iterations), np.uint8), iterations=1)
    # cv2.imshow('dialted', dilated)
    #dilated = cv2.blur(dilated, (80,80))
    #dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    #dilated = fill_holes(dilated)
    print(dilated)
    #cv2.imshow('dialted', dilated)

    # segmented_image_show = cv2.resize(segmented_image, (1080, 1920))
    # res_show = cv2.resize(res, (1080, 1920))
    #res = cv2.bitwise_and(image, image, mask=dilated)
    # cv2.imshow('Segmented', segmented_image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Originale', image)
    # cv2.imshow('Mask', mask)
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    # cv2.imshow('New', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dilated
    # show the image


# def segmentation(image):
#
#     img = cv2.imread(image)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     twoDimage = img.reshape((-1, 3))
#     twoDimage = np.float32(twoDimage)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     K = 2
#     attempts = 10
#     ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     result_image = res.reshape((img.shape))
#     # show the image
#     plt.imshow(result_image)
#     plt.show()



def sharpen(image,number_of_iterations):
    #number_of_iterations = number_of_iterations**3
    kernel = np.array([[-1, -1, -1], [-1, number_of_iterations, -1], [-1, -1, -1]])
    if number_of_iterations != 0:
        dst = cv2.filter2D(image, -1, kernel)
    else:
        dst = image

    return dst

def bilateral_filter(image,number_of_iterations):
    number_of_iterations = number_of_iterations**3
    if number_of_iterations != 0:
        dst = cv2.bilateralFilter(image, 15, number_of_iterations, number_of_iterations)
    else:
        dst = image

    return dst

def blur(image,number_of_iterations):
    number_of_iterations = number_of_iterations
    if number_of_iterations != 0:
        kernel = (number_of_iterations, number_of_iterations)
        dst = cv2.blur(image, kernel)
    else:
        dst = image

    return dst

def compute_distance_from_selected(index, selected_index):
    return abs(index-selected_index)

def process_photos_from_selected(frame_folder, selected_photo, bounce=True):
    frame_folder_list = glob.glob(f"{frame_folder}/*.jpg")
    selected_photo = frame_folder + '\\' + selected_photo
    print(frame_folder_list)
    if (selected_photo) not in frame_folder_list:
        raise ValueError("Selected photo not in the burst photo list")
    selected_photo_id = frame_folder_list.index(selected_photo)
    print(frame_folder_list)
    new_frame_folder_list = []
    previous = cv2.imread(frame_folder_list[0], 1)
    for index, photo in enumerate(frame_folder_list):
        frame = blur(cv2.imread(photo, 1), compute_distance_from_selected(index, selected_photo_id))


        # image = cv2.imread(photo, 1)
        # mask = segmentation(photo, compute_distance_from_selected(index, selected_photo_id))
        # image = cv2.bitwise_and(image, image, mask=mask)
        # #image_previous = cv2.bitwise_and(previous, previous, mask=cv2.bitwise_not(mask))
        # #dst = cv2.add(image, image_previous)
        # #image_previous = dst
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame_folder_list.append(frame_rgb)

    if bounce is True:
        new_frame_folder_list = list(new_frame_folder_list) + list(np.flip(new_frame_folder_list, 0))

    imageio.mimsave('my_awesome.gif', new_frame_folder_list, fps=24)


if __name__ == "__main__":
    process_photos_from_selected('./test', '390.jpg')
    #segmentation('./render/0.jpg', 15)
    #segmentation('./burst/IMG_20220416_161416_015.jpg')

