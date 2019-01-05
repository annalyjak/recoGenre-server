from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras import backend as K
from math import floor
import librosa
import matplotlib.pyplot as plt
import numpy as np
import time
from pydub import AudioSegment
from flask import Flask


#AUDIO PROCESSOR:
def change_3gp_to_mp3(fileName):
    output_Path = fileName + ".mp3"
    AudioSegment.from_file(fileName).export( output_Path, format="mp3")
    return output_Path


def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.

    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def compute_melgram_multiframe(audio_path, all_song=True):
    ''' Compute a mel-spectrogram in multiple frames of the song and returns it in a shape of (N,1,96,1366), where
    96 == #mel-bins, 1366 == #time frame, and N=#frames

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.

    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    if all_song:
        DURA_TRASH = 0
    else:
        DURA_TRASH = 20

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    n_sample_trash = int(DURA_TRASH*SR)

    # remove the trash at the beginning and at the end
    src = src[n_sample_trash:(n_sample-n_sample_trash)]
    n_sample=n_sample-2*n_sample_trash


    # print n_sample
    # print n_sample_fit

    ret = np.zeros((0, 1, 96, 1366), dtype=np.float32)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
        logam = librosa.logamplitude
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS)**2,
                    ref_power=1.0)
        ret = ret[np.newaxis, np.newaxis, :]

    elif n_sample > n_sample_fit:  # if too long
        N = int(floor(n_sample/n_sample_fit))

        src_total=src

        for i in range(0, N):
            src = src_total[(i*n_sample_fit):(i+1)*(n_sample_fit)]

            logam = librosa.logamplitude
            melgram = librosa.feature.melspectrogram
            retI = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                n_fft=N_FFT, n_mels=N_MELS)**2,
                        ref_power=1.0)
            retI = retI[np.newaxis, np.newaxis, :]

            # print retI.shape

            ret = np.concatenate((ret, retI), axis=0)

    return ret


#Functions:
K.set_image_dim_ordering('th')


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def MusicTaggerCRNN(weights='msd', input_tensor=None):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/) to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (96, 1366, 1)
    else:
        input_shape = (1, 96, 1366)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', trainable=False)(x)
    x = Dropout(0.1, name='dropout1', trainable=False)(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2', trainable=False)(x)
    x = Dropout(0.1, name='dropout2', trainable=False)(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', trainable=False)(x)
    x = Dropout(0.1, name='dropout3', trainable=False)(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', trainable=False)(x)
    x = Dropout(0.1, name='dropout4', trainable=False)(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
        x = Dense(10, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)

        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('final_drop')
        preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        return model


#UTILS
def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    save_result_file(sorted_result)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print(name, ':', '%5.3f  ' % score, '   ',)
    print
    return sorted_result


def save_result_file(sorted_result):
    file = open('result.txt', 'w')
    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        file.write(name + ':' + '%5.3f' % score + ';')
    file.close()


def predict_label(preds):
    labels=preds.argsort()[::-1]
    return labels[0]


# Melgram computation
def extract_melgrams(list_path, MULTIFRAMES, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    song_paths = open(list_path, 'r').read().splitlines()
    labels = list()
    num_frames_total = list()
    for song_ind, song_path in enumerate(song_paths):
        print(song_path)
        song_path = change_3gp_to_mp3(song_path)
        if MULTIFRAMES:
            melgram = compute_melgram_multiframe(song_path, process_all_song)
            num_frames = melgram.shape[0]
            num_frames_total.append(num_frames)
            print ('num frames:', num_frames)
            if num_songs_genre != '':
                index = int(floor(song_ind/num_songs_genre))
                for i in range(0, num_frames):
                    labels.append(index)
            else:
                pass
        else:
            melgram = compute_melgram(song_path)

        melgrams = np.concatenate((melgrams, melgram), axis=0)
    if num_songs_genre != '':
        return melgrams, labels, num_frames_total
    else:
        return melgrams, num_frames_total


# Parameters to set
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


# Errors here:
def init_model():
    # Initialize model
    model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if LOAD_WEIGHTS:
        model.load_weights(weights_path + 'crnn_net_gru_adam_ours_epoch_40.h5')
    return model


def main_body():
    model = init_model()

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


def change_to_json(sorted_result):
    ziped = []
    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        ziped.append((name, '%5.3f  ' % score))
    import json
    json = json.dumps([{'genre': genre, 'pred': pred} for genre, pred in ziped])
    return json


# plot with Genres percentage classification
# colors = ['b','g','c','r','m','k','y','#ff1122','#5511ff','#44ff22']
# fig, ax = plt.subplots()
# index = np.arange(tags.shape[0])
# opacity = 1
# bar_width = 0.2
# print(mean)
# #for g in range(0, tags.shape[0]):
# plt.bar(left=index, height=mean, width=bar_width, alpha=opacity, color=colors)
#
# plt.xlabel('Genres')
# plt.ylabel('Percentage')
# plt.title('Scores by genre')
# plt.xticks(index + bar_width / 2, tags)
# plt.tight_layout()
# fig.autofmt_xdate()
# plt.savefig('genres_prediction.png')


# REST API:
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_places():
    sorted_result = main_body()
    json = change_to_json(sorted_result)
    return json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60022, debug=True)
