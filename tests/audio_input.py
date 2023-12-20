# apt install portaudio19-dev : нужно установить если не работает
# sudo apt-get install gcc
import pyaudio
import wave
import numpy as np
import time
import torch
import onnxruntime as ort
from WaveformToSpectrogram import WaveformToSpectrogram as WTS
KNOWN_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "background"
]

class AudioInput:
    def __init__(
            self, 
            sleep_time=10, 
            save=True, 
            ort_session = None
        ):
        self.sleep_time = sleep_time  # сколько секунд нужно записать
        self.save = save  # сохранить ли файл
        # Setup channel infot
        self.FORMAT = pyaudio.paInt16  # data type format
        self.CHANNELS = 1  # number of channels
        self.RATE = 16000  # Sample Rate
        self.CHUNK = 4000  # Block Size
        self.FRAMES = 16000
        #
        self.p = pyaudio.PyAudio()
        self.full_data = torch.empty(0, dtype=torch.float16)
        #-----Отдел модели-------
        self.ort_session = ort_session
        if ort_session is not None:
            self.input_name = self.ort_session.get_inputs()[0].name
        #-----Отдел предобработки звука в мелграмму-----
        self.wts = WTS(
            n_fft=320,
            n_mels=40,
            hop_length=161
        )

    def callback(self, in_data, frame_count, time_info, flag):
        self.full_data = torch.cat((self.full_data, torch.tensor(np.frombuffer(in_data, dtype=np.float32))))
        if len(self.full_data) >= self.FRAMES:
            audio_data = self.full_data[-self.FRAMES:].unsqueeze(0)
            #
            # вызов модуля для тензора audio_data
            print(audio_data.shape)
            wave = self.wts.preprocess_waveform(audio_data)
            print(wave.shape)
            wave = wave.unsqueeze(0)
            print(wave.shape)
            class_answer = make_model_answer(
                input_sample=wave,
                input_name=self.input_name,
                ort_session=self.ort_session
            )
            print(class_answer, KNOWN_COMMANDS[class_answer])
            print("-----")
            #
        return in_data, pyaudio.paContinue

    def save_input(self):
        wave_filename = time.strftime("%b_%d_%Y_%H-%M-%S", time.localtime()) + '_test.wav'
        wf = wave.open(wave_filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.full_data.numpy()))
        wf.close()

    def record(self):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK,
                             stream_callback=self.callback)

        stream.start_stream()
        print("Stream in progress...")

        while stream.is_active():
            time.sleep(self.sleep_time)
            stream.stop_stream()
            print("Stream is stopped.")

        stream.close()
        if self.save:
            self.save_input()
        self.p.terminate()

def make_model_answer(input_sample, input_name, ort_session):
    ort_inputs = {input_name: input_sample.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    tensor_outputs = torch.tensor(ort_outs)[:, :,-1,:].squeeze()
    return torch.argmax(tensor_outputs)

def start_record_and_answer():
    # Загрузка onnx модели
    save_onnx = "final_weights/model.onnx"
    ort_session = ort.InferenceSession(save_onnx)

    # Класс по прослушке микрофона
    a = AudioInput(sleep_time=4, save=True, ort_session=ort_session)

    # Запись
    a.record()


if __name__ == '__main__':
    start_record_and_answer()