    # !pip install pydub
def predict(path):
    import librosa
    import numpy as np
    def noise(data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(data, rate=0.8):
        return librosa.effects.time_stretch(y=data, rate=rate)

    def shift(data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

    # taking any example and checking for techniques.

    data, sample_rate = librosa.load(path)

    def extract_features(data):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
        return result

    def get_features(path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = extract_features(data)
        result = np.array(res1)
        
        # data with noise
        noise_data = noise(data)
        res2 = extract_features(noise_data)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = stretch(data)
        data_stretch_pitch = pitch(new_data, sample_rate)
        res3 = extract_features(data_stretch_pitch)
        result = np.vstack((result, res3)) # stacking vertically
        
        return result

    # todo: convert m4a to wav
    import pickle
    scaler = pickle.load(open('../scaler.pickle', 'rb'))
    encoder = pickle.load(open('../encoder.pickle', 'rb'))

    john_test_recording_features = get_features(path)

    from keras.models import load_model
    import librosa
    import numpy as np

    # convert m4a to wav
    john_test_recording = "./output.wav"

    john_test_recording_features = get_features(john_test_recording)

    john_test_recording_features = scaler.transform(john_test_recording_features)
    john_test_recording_features = np.expand_dims(john_test_recording_features, axis=2)
    model = load_model('../complete_model.h5')

    pred_test = model.predict(john_test_recording_features)
    y_pred = encoder.inverse_transform(pred_test) 

    return(np.unique(y_pred)[0])