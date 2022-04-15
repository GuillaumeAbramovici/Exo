from pydub import AudioSegment
from math import floor
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import cv2
# audio = "./audio/131652__ecfike__grumpy-old-man-3.wav"
# # Load files
# audio_segment = AudioSegment.from_file(audio)
# # Print attributes
# print(f"Channels: {audio_segment.channels}")
# print(f"Sample width: {audio_segment.sample_width}")
# print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
# print(f"Frame width: {audio_segment.frame_width}")
# print(f"Length (ms): {len(audio_segment)}")
# print(f"Frame count: {audio_segment.frame_count()}")
# print(f"Intensity: {audio_segment.dBFS}")
#
#
# # Open wav file and read frames as bytes
# sf_filewave = wave.open('./audio/131652__ecfike__grumpy-old-man-3.wav', 'r')
# signal_sf = sf_filewave.readframes(-1)
# # Convert audio bytes to integers
# soundwave_sf = np.frombuffer(signal_sf, dtype='int16')
#
# #ind = np.argpartition(soundwave_sf, -4)[-4:]
# #top4 = soundwave_sf[ind]
# # local_max = find_peaks(soundwave_sf, threshold=5000)
# # print(len(local_max[0]))
# #ind[:] = [x / 44100 for x in ind]
# #print(sorted(ind))
# #print(top4)
# # Get the sound wave frame rate
# framerate_sf = sf_filewave.getframerate()
# # Find the sound wave timestamps
# time_sf = np.linspace(start=0,
#                       stop=len(soundwave_sf)/framerate_sf,
#                       num=len(soundwave_sf))
# print(len(time_sf))
# # Set up plot
# f= plt.subplots(figsize=(15, 3))
# # Setup the title and axis titles
# plt.title('Amplitude over Time')
# plt.ylabel('Amplitude')
# plt.xlabel('Time (seconds)')
# # Add the audio data to the plot
# plt.plot(time_sf, soundwave_sf, label='Warm Memories', alpha=0.5)
# plt.legend()
# plt.show()
# #

def get_number_of_frame_from_audio_file(audio_file, fps):
    audio_segment = AudioSegment.from_file(audio_file)
    print(f"Length (ms): {len(audio_segment)}")
    number_of_frames = floor(((len(audio_segment)*fps)/1000))
    print(f"Number of frames: {number_of_frames}")
    return number_of_frames


def get_samples_from_audio(audio_file, number_of_frames):
    sf_filewave = wave.open(audio_file, 'r')
    signal_sf = sf_filewave.readframes(-1)
    # Convert audio bytes to integers
    soundwave_sf = np.frombuffer(signal_sf, dtype='int16')
    local_max = find_peaks(soundwave_sf, threshold=5000)
    # Get the sound wave frame rate
    framerate_sf = sf_filewave.getframerate()
    # Find the sound wave timestamps
    time_sf = np.linspace(start=0,
                          stop=len(soundwave_sf)/framerate_sf,
                          num=len(soundwave_sf))
    soundwave_frames = np.array_split(soundwave_sf, number_of_frames)
    sound_wave_mean_values = [np.max(soundwave_frames[i])*3 for i in range(0, number_of_frames)]
    print(f"Number of frames: {sound_wave_mean_values}")
    return sound_wave_mean_values

# get_samples_from_audio(audio, get_number_of_frame_from_audio_file(audio, 24.0))

def convert_video_to_frames(video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1



