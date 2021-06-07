import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0') or v == None:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def display(model_name, boolJsonFormat):
    # Loading the model jsonformat or complete format
    if boolJsonFormat:
        json_file = open(model_name+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_name+".h5")
    else:
        model = load_model(model_name+".h5")

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Contempted",
                    4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprised"}
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        colors = {"white": (255, 255, 255), "black": (
            0, 0, 0), "red": (0, 0, 255)}
        color = colors["red"]

        for (x, y, w, h) in faces:

            """
            #Adding rectangle and label of emotion around the face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), color, 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            """

            #Adding the emoji on the face
            emojiToDisplay = "faces/{}.png".format(emotion_dict[maxindex])
            emojiOverlay = cv2.imread(emojiToDisplay,-1)
            rows,cols,channels = emojiOverlay.shape
            emojiOverlay = cv2.resize(emojiOverlay,(w,h))
            y1, y2 = y, y + emojiOverlay.shape[0]
            x1, x2 = x, x + emojiOverlay.shape[1]

            alpha_s = emojiOverlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for i in range(0, 3):
                frame[y1:y2, x1:x2, i] = (alpha_s * emojiOverlay[:, :, i] +
                                          alpha_l * frame[y1:y2, x1:x2, i])

        cv2.imshow('Video', cv2.resize(
            frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-json",
                        "--json_format",
                        type=str2bool,
                        help="False by default. If true, will read the model in a 2 files format, one json for the architecture and one h5 file for the weights. If False, model is read as a complete model in a h5 file. Usage: -json or -json < one of 'yes', 'true', 't', 'y', '1' > for True. Nothing or -json < one of 'no', 'false', 'f', 'n', '0' >",
                        nargs="?",
                        const=True,
                        default=False)
    parser.add_argument("-n",
                        "--model_name",
                        type=str,
                        help="Name of the model file. Without extension",
                        nargs="?",
                        default="ferplusModel")

    args = parser.parse_args()
    for i in range(5) :
        print("Press Q to [Q]uit or Ctrl+C for CommandLine Interrupt")
    display(model_name=args.model_name, boolJsonFormat = args.json_format)
