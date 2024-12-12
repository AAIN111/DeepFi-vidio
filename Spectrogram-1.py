import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Step 1: Simulate IoT Signal
def generate_iot_signal(frequency, sampling_rate, duration, snr, label):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, np.sqrt(10**(-snr / 10)), signal.shape)
    return signal + noise, label

# Step 2: Generate WiFi Signal
def generate_wifi_signal(sampling_rate, duration, snr):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    wifi_signal = np.random.choice([1, -1], size=t.shape)
    noise = np.random.normal(0, np.sqrt(10**(-snr / 10)), wifi_signal.shape)
    return wifi_signal + noise

# Step 3: Collision Simulation
def simulate_collision(iot_signal, wifi_signal):
    return iot_signal + wifi_signal

# Step 4: FFT and Spectrogram
def compute_spectrogram(signal, sampling_rate, nfft=256):
    plt.figure(figsize=(8, 6))
    spectrogram, freqs, bins, im = plt.specgram(
        signal, NFFT=nfft, Fs=sampling_rate, noverlap=nfft // 2, cmap='jet'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    # plt.ylim(0,20000)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.10)
    plt.subplots_adjust(wspace=0,hspace=0) 
    plt.show()
    return spectrogram

# Step 5: Data Preparation for CNN
def prepare_data(n_samples, sampling_rate, duration, snr_range):
    classes = ['LoRa', 'ZigBee', 'Bluetooth']
    frequencies = [2.4e3, 2.4e3, 2.4e3]  # Example frequencies for each class
    data = []
    labels = []
    for _ in range(n_samples):
        for freq, label in zip(frequencies, classes):
            snr = np.random.uniform(*snr_range)
            iot_signal, lbl = generate_iot_signal(freq, sampling_rate, duration, snr, label)
            wifi_signal = generate_wifi_signal(sampling_rate, duration, snr)
            collision_signal = simulate_collision(iot_signal, wifi_signal)
            spectrogram = compute_spectrogram(collision_signal, sampling_rate)
            data.append(spectrogram)
            labels.append(lbl)
    return np.array(data), np.array(labels)

# Step 6: Build a Simple CNN
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example Usage
sampling_rate = 1e6  # 1 MHz
duration = 0.01  # 10 ms
snr_range = [10, 20]  # SNR range in dB

# Generate data
n_samples = 10  # Number of samples per class
data, labels = prepare_data(n_samples, sampling_rate, duration, snr_range)

# Preprocess data
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Reshape data for CNN
data = data[..., np.newaxis]  # Add channel dimension

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)

# Build and Train CNN Model
input_shape = X_train.shape[1:]
num_classes = len(le.classes_)
cnn_model = build_cnn_model(input_shape, num_classes)
cnn_model.summary()

# Train Model
history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4)

# Evaluate Model
test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")
