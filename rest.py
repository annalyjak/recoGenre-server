import numpy as np
import time
import tensorflow as ts
from flask import Flask

from data_processing import extract_melgrams
from model import MusicTaggerCRNN
from utils import change_to_json, predict_label, sort_result


TEST = 1

LOAD_MODEL = 0
LOAD_WEIGHTS = 1
MULTIFRAMES = 1
time_elapsed = 0

# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths to set
model_name = "example_model"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

test_songs_list = 'list_example.txt'


def init_model():
    # Initialize model
    global model, graph
    model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    graph = ts.get_default_graph()

    if LOAD_WEIGHTS:
        model.load_weights(weights_path + 'crnn_net_gru_adam_ours_epoch_40.h5')
    return model


def main_body():
    X_test, num_frames_test = extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')

    num_frames_test = np.array(num_frames_test)

    t0 = time.time()

    print('\n--------- Predicting ---------', '\n')

    results = np.zeros((X_test.shape[0], tags.shape[0]))
    predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
    predicted_labels_frames = np.zeros((X_test.shape[0], 1))

    song_paths = open(test_songs_list, 'r').read().splitlines()

    previous_numFrames = 0
    n = 0
    for i in range(0, num_frames_test.shape[0]):
        print('Song number' + str(i) + ': ' + song_paths[i])

        num_frames = num_frames_test[i]
        print('Num_frames of 30s: ', str(num_frames), '\n')

        with graph.as_default():
            results[previous_numFrames:previous_numFrames + num_frames] = model.predict(
                X_test[previous_numFrames:previous_numFrames + num_frames, :, :, :])

        s_counter = 0
        for j in range(previous_numFrames, previous_numFrames + num_frames):
            # normalize the results
            total = results[j, :].sum()
            results[j, :] = results[j, :] / total
            print('Percentage of genre prediction for seconds ' + str(20 + s_counter * 30) + ' to ' \
                  + str(20 + (s_counter + 1) * 30) + ': ')
            sort_result(tags, results[j, :].tolist())

            predicted_label_frames = predict_label(results[j, :])
            predicted_labels_frames[n] = predicted_label_frames
            s_counter += 1
            n += 1

        print('\n', 'Mean genre of the song: ')
        results_song = results[previous_numFrames:previous_numFrames + num_frames]
        mean = results_song.mean(0)
        sorted_result = sort_result(tags, mean.tolist())

        predicted_label_mean = predict_label(mean)

        predicted_labels_mean[i] = predicted_label_mean
        print('\n', 'The predicted music genre for the song is', str(tags[predicted_label_mean]), '!\n')

        previous_numFrames = previous_numFrames + num_frames

        print('************************************************************************************************')
        return sorted_result


# REST API:
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_places():
    sorted_result = main_body()
    json = change_to_json(sorted_result)
    return json


if __name__ == '__main__':
    init_model()
    print("model has been inited")
    app.run(host='0.0.0.0', port=60022, debug=True)
    print("app's ready")
