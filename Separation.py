import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Parameters
Fs = 1000  # Sampling frequency (Hz)
T = 1  # Duration (s)
t = np.linspace(0, T, int(Fs * T), endpoint=False)

# Generate Signals
wifi_signal = np.sin(2 * np.pi * 50 * t)  # WiFi signal at 50 Hz
zigbee_signal = chirp(t, f0=100, f1=150, t1=T, method='linear')  # ZigBee-like chirp
noise = np.random.normal(0, 0.5, size=t.shape)  # Gaussian noise

# Composite Signal with Interference
composite_signal = wifi_signal + zigbee_signal + noise

# # Visualization of Signals
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(t, wifi_signal)
# plt.title("WiFi Signal")
# plt.subplot(3, 1, 2)
# plt.plot(t, zigbee_signal,color='black',linewidth=2,\
#     linestyle='-')
# # plt.ylim([20,100])
# plt.xlim([0,0.2])
# plt.xlabel("Time (ms)",fontsize=20)
# plt.ylabel("Amplitude",fontsize=20)
# plt.tick_params(labelsize=20)  
# plt.subplots_adjust(left=0.20, right=0.95, top=0.97, bottom=0.15)
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.title("ZigBee Signal")
# plt.subplot(3, 1, 3)
plt.plot(t, composite_signal,color='black',linewidth=2,\
    linestyle='-')
plt.xlim([0,0.2])
plt.xlabel("Time (ms)",fontsize=20)
plt.ylabel("Amplitude",fontsize=20)
plt.tick_params(labelsize=20)  
plt.subplots_adjust(left=0.15, right=0.95, top=0.97, bottom=0.15)
plt.subplots_adjust(wspace=0,hspace=0)
# plt.title("Composite Signal with Noise and Interference")
plt.tight_layout()
plt.show()

# Prepare Spectrograms for Model Training
# def generate_spectrogram(signal, label):
#     from scipy.signal import spectrogram
#     f, t, Sxx = spectrogram(signal, Fs, nperseg=128)
#     Sxx = np.log10(Sxx + 1e-10)  # Log scale for better visualization
#     return Sxx, label

# signals = [wifi_signal, zigbee_signal, composite_signal]
# labels = [0, 1, 2]  # 0: WiFi, 1: ZigBee, 2: Composite

# spectrograms = []
# labels_list = []

# for signal, label in zip(signals, labels):
#     for _ in range(50):  # Augment data with slight variations
#         noisy_signal = signal + np.random.normal(0, 0.1, size=signal.shape)
#         Sxx, lbl = generate_spectrogram(noisy_signal, label)
#         spectrograms.append(Sxx)
#         labels_list.append(lbl)

# spectrograms = np.array(spectrograms)
# labels_list = np.array(labels_list)

# # Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels_list, test_size=0.2, random_state=42)
# X_train = X_train[..., np.newaxis]  # Add channel dimension for CNN
# X_test = X_test[..., np.newaxis]
# y_train = to_categorical(y_train, num_classes=3)
# y_test = to_categorical(y_test, num_classes=3)

# Build CNN Model
# Updated CNN Model
# Updated CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 1)),  # Pooling only along the time axis
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 1)),  # Adjust pooling to fit spatial constraints
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

import tensorflow as tf

# Resize spectrograms
X_train_resized = tf.image.resize(X_train, [64, 64])  # Resize spectrograms
X_test_resized = tf.image.resize(X_test, [64, 64])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot Training History
# plt.figure(figsize=(10, 4))
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Visualize Spectrograms
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(spectrograms[0], aspect='auto', cmap='hot')
# plt.title("WiFi Spectrogram")
# plt.tick_params(labelsize=12) 
# plt.xlabel("Time (ms)",fontsize=12)
# plt.ylabel("Frequency(GHz)",fontsize=12) 
# plt.subplot(1, 3, 2)
# plt.imshow(spectrograms[1], aspect='auto', cmap='hot')
# plt.tick_params(labelsize=12)  
# plt.title("ZigBee Spectrogram")
# plt.xlabel("Time (ms)",fontsize=12)
# plt.ylabel("Frequency(GHz)",fontsize=12) 
# plt.subplot(1, 3, 3)
# plt.imshow(spectrograms[2], aspect='auto', cmap='hot')
# plt.title("Composite Spectrogram")
# plt.tick_params(labelsize=12)  
# # # Display the plot
# plt.xlabel("Time (ms)",fontsize=12)
# plt.ylabel("Frequency(GHz)",fontsize=12) 
# plt.subplots_adjust(left=0.06, right=0.95, top=0.97, bottom=0.08)
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.tight_layout()
# plt.show()