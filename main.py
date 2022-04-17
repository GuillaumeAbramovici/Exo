# import the opencv library
import datetime
import os.path
import random
import time
import math
import glob
import cv2
import numpy as np
from pydub import AudioSegment
from Audio import get_number_of_frame_from_audio_file, get_samples_from_audio, convert_video_to_frames


audio = "./audio/test.wav"
# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
frame_size = (frame_width, frame_height)
print(frame_size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 12
save_path = 'timelapse.avi'
video = 'videoplayback.mp4'

seconds_duration = 240
timelapse_img_dir = "road"
now = datetime.datetime.now()
finish_time = now + datetime.timedelta(seconds=seconds_duration)
seconds_between_shots = 0.25
i = 0
# out = cv2.VideoWriter(save_path, fourcc, fps, (1280,720))

#
# ### CAMERA CAPTURE ####
# ret, frame = vid.read()
# first_frame = frame
# previous_frame = first_frame
# while datetime.datetime.now() < finish_time:
#
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#
#     #frame = cv2.addWeighted(previous_frame, 0.5, frame, 0.5, 0)
#
#     filename = f"{timelapse_img_dir}/{i}.jpg"
#     i += 1
#     # Display the resulting frame
#     cv2.imwrite(filename, frame)
#     cv2.imshow('frame', frame)
#     #previous_frame = frame
#     #time.sleep(seconds_between_shots)
#
#     # Exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


def zoom_center(list_img):
    y_size = list_img[0][0].shape[0]
    x_size = list_img[0][0].shape[1]
    zoom_factor = len(list_img)/4
    print(list_img[0])
    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_factor))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_factor))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_factor))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_factor))

    # first crop image then scale
    img_cropped = list_img[0][y1:y2, x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)

def sharp(list_images):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    dst = list_images[0]
    for n in range(len(list_images)):
        if n == 0:
            pass
        else:
            dst = cv2.filter2D(src=dst, ddepth=-1, kernel=kernel)
    return dst

# def blur(image,number_of_iterations):
#     if number_of_iterations != 0:
#         kernel = np.ones((number_of_iterations, number_of_iterations), np.float32) / number_of_iterations*number_of_iterations
#         dst = cv2.filter2D(image, -1, kernel)
#     else:
#         dst = image
#
#     return dst

def medianblur(image,number_of_iterations):
    if number_of_iterations%2 != 1:
        number_of_iterations+=1
    if number_of_iterations != 0:
        dst = cv2.medianBlur(image, number_of_iterations)
    else:
        dst = image

    return dst

def blur(image,number_of_iterations):
    if number_of_iterations != 0:
        kernel = (number_of_iterations, number_of_iterations)
        dst = cv2.blur(image, kernel)
    else:
        dst = image

    return dst


def blend(list_images): # Blend images equally.
    dst = list_images[0]
    kernel_7x7 = np.ones((7, 7), np.float32) / 49
    for n in range(len(list_images)):
        if n == 0:
            pass
        else:
            alpha = 1.0 / (n + 1)
            beta = 1.0 - alpha
            blurred = cv2.filter2D(list_images[n], -1, kernel_7x7)
            dst = cv2.addWeighted(blurred, alpha, dst, beta, 0.0)
    return dst

def number_frames(number, frames):
    return min(frames - number, number)


def timelapse_blend(video_frames, number_of_frames, amplitude_samples):
    """

    :param frames: open images with imread
    :param lapse: duration
    :return:
    """

    # Split video_frames into number_of_frames sub array
    # Multiple sub array of almost the same size
    sub_frames = np.array_split(video_frames, number_of_frames)

    # Mapping of values from amplitude to the number of frames of each sub array
    amplitude_min_max = [min(amplitude_samples), max(amplitude_samples)]
    max_len_sub_array = [0, len(sub_frames[0])]
    mapped_values = np.interp(amplitude_samples, xp=amplitude_min_max, fp=max_len_sub_array)
    mapped_values_ceil = [math.ceil(x) for x in mapped_values]
    print(mapped_values_ceil)
    # Arrays
    chunks = []
    blended_images = []
    #print(len(sub_frames[0]))
    for i, frames in enumerate(sub_frames):

        # Generate random sizes for frame blending
        # half_size_samples = random.sample(range(1, 5), 1)[0]
        bottom_chunk = max(0, math.ceil(len(frames)/2)-math.ceil(mapped_values_ceil[i]))
        ceil_chunk = min(len(video_frames), math.ceil(len(frames)/2)+math.ceil(mapped_values_ceil[i]))
        if bottom_chunk == ceil_chunk:
            ceil_chunk += 1
        print(f"Number: {i}")
        print(f"Bottom Chunk: {bottom_chunk}")
        print(f"Ceil Chunk: {ceil_chunk}")
        chunk = frames[bottom_chunk:ceil_chunk,:,:,:]
        chunks.append(chunk)
    #print(chunks)
    for chunk in chunks:
        #print(chunk[0])
        blended_images.append(blend(chunk))

    return blended_images


def fill_chunk(chunk, maxlen):
    nb_elements_frame = len(chunk)
    mini_chunk_size = math.floor(maxlen/nb_elements_frame)
    new_chunk = []
    for frame in chunk:
        for id in range(0, mini_chunk_size):
            new_chunk.append(frame)
            if len(new_chunk) > maxlen:
                break
    return new_chunk


def timelapse_fps(video_frames, number_of_frames, amplitude_samples):
    """

    :param frames: open images with imread
    :param lapse: duration
    :return:
    """

    # Split video_frames into number_of_frames sub array
    # Multiple sub array of almos the same size
    sub_frames = np.array_split(video_frames, number_of_frames)

    # Mapping of values from amplitude to the number of frames of each sub array
    amplitude_min_max = [min(amplitude_samples), max(amplitude_samples)]
    max_len_sub_array = [0, len(sub_frames[0])]
    mapped_values = np.interp(amplitude_samples, xp=amplitude_min_max, fp=max_len_sub_array)
    mapped_values_ceil = [math.ceil(x) for x in mapped_values]
    print(mapped_values_ceil)
    # Arrays
    chunks = []
    fps_images = []
    #print(len(sub_frames[0]))
    for i, frames in enumerate(sub_frames):

        # Generate random sizes for frame blending
        # half_size_samples = random.sample(range(1, 5), 1)[0]
        bottom_chunk = max(0, math.ceil(len(frames)/2)-math.ceil(mapped_values_ceil[i]))
        ceil_chunk = min(len(video_frames), math.ceil(len(frames)/2)+math.ceil(mapped_values_ceil[i]))
        if bottom_chunk == ceil_chunk:
            ceil_chunk += 1
        # print(f"Number: {i}")
        # print(f"Bottom Chunk: {bottom_chunk}")
        # print(f"Ceil Chunk: {ceil_chunk}")
        chunk = frames[bottom_chunk:ceil_chunk,:,:,:]
        for index, image in enumerate(chunk):
            chunk[index]=medianblur(image, ceil_chunk-bottom_chunk)
        maxlen_chunk = fill_chunk(chunk, len(sub_frames[0]))
        #if len(chunk) < len(sub_frames):

        chunks.append(maxlen_chunk)
        #print(len(chunk))

    for chunk in chunks:
        for frame in chunk:
            fps_images.append(frame)

    return fps_images, len(sub_frames[0])


def images_to_video(out, blended_images, clear_images = False):
    for file in blended_images:
        out.write(file)

    if clear_images:
        for file in image_list:
            os.remove(file)
    # After the loop release the cap object


# convert_video_to_frames(video)
fps = 6
number_of_frames = get_number_of_frame_from_audio_file(audio, fps)
amplitude_samples = get_samples_from_audio(audio, get_number_of_frame_from_audio_file(audio, fps))
image_list = glob.glob(f"{timelapse_img_dir}/*.jpg")
sorted_images = sorted(image_list, key=os.path.getmtime)
print(sorted_images)
images_sorted_open = []
for my_file in sorted_images:
    this_image = cv2.imread(my_file, 1)
    images_sorted_open.append(this_image)

blended_images, fps_out = timelapse_fps(images_sorted_open, number_of_frames, amplitude_samples)
print(f"fps: {fps}")
print(f"FPS out: {fps_out}")
# print(type(blended_images))
fps_final = fps_out*fps
print(f"fps final: {fps_final}")
out = cv2.VideoWriter(save_path, fourcc, fps_final, (1280,720))
images_to_video(out, blended_images)
# os.remove(save_path)
# os.remove('output.mp4')

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



#
# def images_to_video(out, img_dir, clear_images=False):
#     image_list = glob.glob(f"{img_dir}/*.jpg")
#     sorted_images = sorted(image_list, key=os.path.getmtime)
#     for file in sorted_images:
#         image_frame = cv2.imread(file)
#         out.write(image_frame)
#     if clear_images:
#         for file in image_list:
#             os.remove(file)
#     # After the loop release the cap object
# def timelapse_blend(frames, lapse):
#     """
#
#     :param frames: open images with imread
#     :param lapse: duration
#     :return:
#     """
#
#     # Compute number of frames
#     lapse_frames = fps*lapse
#     number_of_frames = math.floor(len(frames)/lapse_frames)
#
#     # Generate random number for random frames to generate the timelapse frame around
#     random_numbers = sorted(random.sample(range(1, len(frames)), number_of_frames))
#
#     # Arrays
#     chunks = []
#     blended_images = []
#
#     for number in random_numbers:
#
#         # Generate random sizes for frame blending
#         half_size_samples = random.sample(range(1, 5), 1)[0]
#
#         bottom_chunk = max(0, number-half_size_samples)
#         ceil_chunk = min(len(frames), number+half_size_samples)
#         chunk = frames[bottom_chunk:ceil_chunk]
#         chunks.append(chunk)
#
#     for chunk in chunks:
#         blended_images.append(blend(chunk))
#
#     return blended_images