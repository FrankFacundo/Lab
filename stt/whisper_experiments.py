import matplotlib.pyplot as plt
import numpy as np
import whisper

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("the_big_short.mp3")
print(type(audio))
print(audio.shape)
# <class 'numpy.ndarray'>
# (1857792,)

audio = whisper.pad_or_trim(audio)
print(type(audio))
print(audio.shape)
# <class 'numpy.ndarray'>
# (480000,)

# Plot the audio waveform
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(audio) / 16000, num=len(audio)), audio)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
print(mel.shape)
# torch.Size([80, 3000])

# Plot the log-Mel spectrogram
mel_np = mel.cpu().detach().numpy()
plt.figure(figsize=(10, 4))
plt.imshow(mel_np, aspect="auto", origin="lower")
plt.title("Log-Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.show()

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
