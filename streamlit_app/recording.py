import pyaudio
import wave
def record():
    # Define parameters for audio recording
    FORMAT = pyaudio.paInt16  # Sample format
    CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 44100              # Sample rate (samples per second)
    RECORD_SECONDS = 5        # Duration of recording in seconds
    OUTPUT_FILENAME = "output.wav"  # Output WAV file name

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a stream for recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=1024)

    print("Recording...")

    frames = []

    # Record audio in chunks and store them in the frames list
    for _ in range(0, int(RATE / 1024 * RECORD_SECONDS)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    # Close and terminate the audio stream and PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_FILENAME
