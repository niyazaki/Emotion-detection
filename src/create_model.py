import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_model_history(model_history, model_name):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy'])+1),
                model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy'])+1),
                model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy'])+1),
                      len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(model_name.split(".")[0]+".png")
    plt.show()


def createModel(model_name, train_dir, val_dir, batch_size, num_epoch):
    emotions = os.listdir(val_dir)
    # number of emotions = the number of folders in data/train or data/test
    output_size = len(emotions)
    num_train = sum([len(os.listdir(os.path.join(train_dir, emotion)))
                     for emotion in emotions])
    num_val = sum([len(os.listdir(os.path.join(val_dir, emotion)))
                   for emotion in emotions])
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    # If you want to train the same model or try other models, go for this
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    plot_model_history(model_info, model_name)
    model.save(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--model_name",
                        type=str,
                        help="Name of the output (model) file. Don't forget the \".h5\" extension",
                        required=True)

    parser.add_argument("-td",
                        "--train_dir",
                        type=str,
                        help="Path to the main folder with the training dataset",
                        nargs="?",
                        default="data/train")

    parser.add_argument("-vd",
                        "--val_dir",
                        type=str,
                        help="Path to the main folder with the validating (test) dataset",
                        nargs="?",
                        default="data/test")

    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        help="Number of elements in batch, default is 64",
                        nargs="?",
                        default=64)

    parser.add_argument("-e",
                        "--num_epoch",
                        type=int,
                        help="Number of epoch for the model, default is 30.",
                        nargs="?",
                        default=30)

    args = parser.parse_args()

    createModel(model_name=args.model_name, train_dir=args.train_dir,
                val_dir=args.val_dir, batch_size=args.batch_size, num_epoch=args.num_epoch)
