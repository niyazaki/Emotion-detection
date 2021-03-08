#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import os
import csv
import argparse
import numpy as np
from itertools import islice
from PIL import Image

# List of folders for training, validation and test.
folder_names = {'Training': 'Train',
                'PublicTest': 'Valid',
                'PrivateTest': 'Test'}

emotions = ["neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"]


def slice_to_emotion(emotion_slice):
    '''Convert the emotion values of FER+ to an emotion using the majority selection'''
    emotion_slice = [int(i) for i in emotion_slice]
    #sum_list = sum(emotion_slice)
    maxval = max(emotion_slice)
    #emotion = "unknown"
    # if maxval >= 0.5*sum_list:
    emotion = emotions[emotion_slice.index(maxval)]
    return emotion


def str_to_image(image_blob):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(image_data)


def main(base_folder, fer_path, ferplus_path):
    '''
    Generate PNG image files from the combined fer2013.csv and fer2013new.csv file. The generated files
    are stored in their corresponding folder for the trainer to use.

    Args:
        base_folder(str): The base folder that contains  'FER2013Train', 'FER2013Valid' and 'FER2013Test'
                          subfolder.
        fer_path(str): The full path of fer2013.csv file.
        ferplus_path(str): The full path of fer2013new.csv file.
    '''

    print("Start generating ferplus images.")

    for _, value in folder_names.items():
        folder_path = os.path.join(base_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            for emotion in emotions:
                emotion_path = os.path.join(base_folder, value, emotion)
                if not os.path.exists(emotion_path):
                    os.makedirs(emotion_path)

    ferplus_entries = []
    with open(ferplus_path, 'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)

    index = 0
    with open(fer_path, 'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                image = str_to_image(row[1])
                # Emotion part without NF and unknown
                emotion_slice = ferplus_row[2:-2]
                emotion = slice_to_emotion(emotion_slice)
                image_path = os.path.join(
                    base_folder, folder_names[row[2]], emotion, file_name)
                image.save(image_path, compress_level=0)
            index += 1

    print("Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--base_folder",
                        type=str,
                        help="Base folder containing the training, validation and testing folder.",
                        required=True)
    parser.add_argument("-fer",
                        "--fer_path",
                        type=str,
                        help="Path to the original fer2013.csv file.",
                        required=True)

    parser.add_argument("-ferplus",
                        "--ferplus_path",
                        type=str,
                        help="Path to the new fer2013new.csv file.",
                        required=True)

    args = parser.parse_args()
    main(args.base_folder, args.fer_path, args.ferplus_path)
