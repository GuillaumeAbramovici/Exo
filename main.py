# import the opencv library
import datetime
import os.path
import time
import glob
import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
frame_size = (frame_width, frame_height)
print(frame_size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
save_path = 'timelapse.avi'


seconds_duration = 10
timelapse_img_dir = "render"
now = datetime.datetime.now()
finish_time = now + datetime.timedelta(seconds=seconds_duration)
seconds_between_shots = 1
i = 0
out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

# ret, frame = vid.read()
# first_frame = frame
# previous_frame = first_frame
while datetime.datetime.now() < finish_time:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #frame = cv2.addWeighted(previous_frame, 0.5, frame, 0.5, 0)

    filename = f"{timelapse_img_dir}/{i}.jpg"
    i += 1
    # Display the resulting frame
    cv2.imwrite(filename, frame)
    cv2.imshow('frame', frame)
    #previous_frame = frame
    #time.sleep(seconds_between_shots)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

clear_images = True


def blend(list_images): # Blend images equally.
    dst = list_images[0]
    for n in range(len(list_images)):
        if n == 0:
            pass
        else:
            alpha = 1.0 / (n + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(list_images[n], alpha, dst, beta, 0.0)
    return dst


def timelapse_blend(frames, lapse):
    lapse_frames = fps*lapse
    chunks = []
    chunks = np.array_split(frames, lapse_frames)
    blended_images = []
    for chunk in chunks:
        blended_images.append(blend(chunk))
    return blended_images


def images_to_video(out, blended_images):
    for file in blended_images:
        #image_frame = cv2.imread(file)
        out.write(file)


image_list = glob.glob(f"{timelapse_img_dir}/*.jpg")
sorted_images = sorted(image_list, key=os.path.getmtime)
images_sorted_open = []
for my_file in sorted_images:
    this_image = cv2.imread(my_file, 1)
    images_sorted_open.append(this_image)
blended_images = timelapse_blend(images_sorted_open, seconds_between_shots)
images_to_video(out, blended_images)


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
