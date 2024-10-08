import base64
import requests
import pdb 
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy import signal
import librosa

# OpenAI API Key
api_key = open("key.txt").read().strip()
system_prompt_audio = """
		You are an expert in understanding spectogram. You can understand the dominant frequencies and signal characteristics by inspecting frequency. To do so, you will: 

		1) Look for horizontal lines or regions that stand out with brighter colors or higher intensity compared to the surrounding areas.
		2) Locate the dominant horizontal line or band that catches your attention.
		3) Trace the line horizontally to the y-axis to determine its corresponding frequency.

		You can clearly describe a plot in natural language. Now, I am going to give a plot of a temporal signal to you.
		"""
system_prompt_ecg = """
You are an expert in perform spectrum analysis. Answer my question directly by inspecting plots.
When look at the plot, be careful on retreiving the numbers from axes.
"""
# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def image_query(image_path, system_prompt, query, api_key):
	# Getting the base64 string
	base64_image = encode_image(image_path)
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}
	payload = {
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": system_prompt}, 
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": query
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{base64_image}"
						}
					}
				]
			}
		],
		"max_tokens": 512
	}
	response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
	message = response.json()['choices'][0]['message']['content']

	return message

def inspect_fft(data, query, fs):
	# Path to your image
	image_path = './vis/' + data.split('/')[-2] + '_' + data.split('/')[-1].split('.')[0] + '_fft.png'
	
	if 'npy' in data:
		ecg_signal = np.load(data, allow_pickle=True)
	elif 'wav' in data:
		ecg_signal, sr = librosa.load(data)
		fs = sr
	
	# Compute the Fast Fourier Transform (FFT)
	fft_result = np.fft.fft(ecg_signal)

	# Compute the frequency axis
	n = len(fft_result)
	frequency = np.fft.fftfreq(n, d=1/fs)

	# Compute the magnitude spectrum
	magnitude = np.abs(fft_result) / n

	tick_locations = np.linspace(0, np.max(frequency[:n // 2]), num=10)  # Adjust `num` for desired density
	tick_labels = np.round(tick_locations, 2)

	# Plot the magnitude spectrum
	plt.figure(figsize=(10, 6))
	plt.plot(frequency[:n // 2], magnitude[:n // 2] * 2)  # Multiply by 2 for single-sided spectrum
	plt.title('Magnitude Spectrum of the Signal')
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude')
	# plt.xticks(tick_locations, tick_labels)
	plt.grid()
	plt.tight_layout()
	plt.savefig(image_path)
	system_prompt = system_prompt_ecg

	pdb.set_trace()
	message = image_query(image_path, system_prompt, query, api_key)
	
	print(message)
	# pdb.set_trace()
	return message

def inspect_ts(data, query, fs=None):
	# Path to your image
	image_path = './vis/' + data.split('/')[-2] + '_' + data.split('/')[-1].split('.')[0] + '_ts.png'
	
	if 'npy' in data:
		ecg_signal = np.load(data)
	elif 'wav' in data:
		ecg_signal, sr = librosa.load(data)
		fs = sr

	# Compute the frequency axis
	n = len(ecg_signal)
		
	if fs != None:
		time_duration = np.linspace(0, n/fs, n)
	else:
		time_duration = np.arange(0, n)

	# Plot the magnitude spectrum
	plt.figure(figsize=(10, 6))

	plt.plot(time_duration, ecg_signal)  # Multiply by 2 for single-sided spectrum
	plt.title('Time-series Signal')
	if fs != None:
		plt.xlabel('Second (Hz)')
	else:
		plt.xlabel('Index')
	plt.ylabel('Value')
	plt.grid()
	plt.tight_layout()
	plt.savefig(image_path)
	system_prompt = system_prompt_ecg

	message = image_query(image_path, system_prompt, query, api_key)
	
	print(message)
	# pdb.set_trace()
	return message

def inspect_spectrogram(data, query):
	# Path to your image
	image_path = './vis/' + data.split('/')[-2] + '_' + data.split('/')[-1].split('.')[0] + '_spect.png'
	fmax = 8192
	if 'wav' in data:
		# Load the audio file
		y, sr = librosa.load(data)

		# Compute the Mel spectrogram
		S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=fmax)
		# Display the Mel spectrogram
		plt.figure(figsize=(10, 10))
		librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=fmax, x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Mel Spectrogram')
		plt.tight_layout()
		plt.savefig(image_path)
		system_prompt = system_prompt_audio

	elif 'npy' in data:
		fs = 500
		ecg_signal = np.load(data)
		# Compute the Fast Fourier Transform (FFT)
		fft_result = np.fft.fft(ecg_signal)

		# Compute the frequency axis
		n = len(fft_result)
		frequency = np.fft.fftfreq(n, d=1/fs)

		# Compute the magnitude spectrum
		magnitude = np.abs(fft_result) / n
		
		# Plot the magnitude spectrum
		plt.figure(figsize=(10, 6))
		plt.plot(frequency[:n // 2], magnitude[:n // 2] * 2)  # Multiply by 2 for single-sided spectrum
		plt.title('Magnitude Spectrum of the Signal')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Magnitude')
		plt.grid()
		plt.tight_layout()
		plt.savefig(image_path)
		system_prompt = system_prompt_ecg

	message = image_query(image_path, system_prompt, query, api_key)

	print(message)
	# pdb.set_trace()
	return message

if __name__ == "__main__":
	# inspect('./benchmark/speech-Siren/1.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing? What are the frequency ranges of the speech and noise?")
	# inspect('./benchmark/noise_sample/TelephoneRing1.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing?")
	# inspect('./benchmark/noise_sample/Siren.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing?")
	# inspect_spectrogram('./benchmark/speech-TelephoneRing1/1.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing? What are the frequency ranges of the speech and noise?")
	# inspect_spectrogram('./llm_response/gpt-4o_speech-TelephoneRing1_2_3.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing? What are the frequency ranges of the speech and noise?")
	# inspect_fft('./llm_response/gpt-4o_speech-TelephoneRing1_2_3.wav', "Can you describe the frequency and identify if there are any noise artifacts, especially phone ringing? What are the frequency ranges of the speech and noise?", 8000)
	# inspect_fft('./benchmark/ecg_data-powerline_2/1.npy', "This is an ECG spectrogram. Is there any frequency correspond to noise?", 500)
	# inspect_ts('./benchmark/ecg_data-powerline_2/1_gt.npy', "What does this time series typically look like?", fs=500)

	# inspect_fft('./benchmark/ecg_data-powerline_2/1.npy', query="Inspect the frequency spectrum of the ECG data to identify the presence of powerline noise.", fs=500)
	inspect_fft(data='./llm_response/gpt-4o_ecg_data-powerline_2_2_5.npy', query='Has the 50 Hz and its first harmonic 100 Hz noise been removed from this ECG signal?', fs=500)
	# inspect_ts('./benchmark/speech-TelephoneRing2/1.wav', "What's in the image?")