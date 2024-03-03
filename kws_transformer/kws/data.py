from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
# from torchaudio.datasets import SPEECHCOMMANDS
# import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
# from nnAudio.features.mel import MelSpectrogram
# import matplotlib.pyplot as plt
# import einops
# from nnAudio.features.mel import MFCC
from typing import Literal
# from torch.nn.utils.rnn import pad_sequence
import lightning as L
# from kws.preprocessing_data.preproccesing import SpeechCommandsData
import torchaudio
from collections import defaultdict

KNOWN_COMMANDS = ["yes",
                  "no",
                  "up",
                  "down",
                  "left",
                  "right",
                  "on",
                  "off",
                  "stop",
                  "go",
                  "background"]

# class SubsetSC(SPEECHCOMMANDS):
#     def __init__(self, dest, subset: str = None):
#         super().__init__(dest, download=True)

#         def load_list(filename):
#             filepath = os.path.join(self._path, filename)
#             with open(filepath) as fileobj:
#                 return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#         elif subset == "testing":
#             self._walker = load_list("testing_list.txt")
#         elif subset == "training":
#             excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
#             excludes = set(excludes)
#             self._walker = [w for w in self._walker if w not in excludes]

# class AudioDataset(Dataset):
#     def __init__(
#             self, 
#             destination, 
#             set_type:Literal["training", "validation", "testing"] | None, 
#             audio_rate:int=16000,
#             labels = None
#         ) -> None:
#         super().__init__()
#         self.audio_set = SubsetSC(dest=destination, subset=set_type)
#         self.audio_rate = audio_rate

#         if labels is None:
#             self.labels = sorted(list(set(datapoint[2] for datapoint in self.audio_set)))
#         else:
#             self.labels = labels
    
#     def __len__(self):
#         return len(self.audio_set)

#     def ensure_lenght(self, wave):
#         if wave.shape[1] != self.audio_rate:
#             wave = torch.cat((wave, torch.zeros((1, self.audio_rate - wave.shape[1]))), dim=1)
#         return wave
    
#     def key_translate(self, label_name):
#         return torch.tensor(self.labels.index(label_name))

#     def __getitem__(self, index):
#         waveform, sample_rate, label, speaker_id, utterance_number = self.audio_set[index]
#         waveform = self.ensure_lenght(waveform)
#         label = self.key_translate(label)
#         return waveform, label

# class Audio_DataModule_OLD(L.LightningDataModule):
#     def __init__(self, data_destination, batch_size=32, audio_rate:int=16000, labels=None) -> None:
#         super().__init__()

#         self.destination = data_destination
#         self.batch_size = batch_size
#         self.audio_rate = audio_rate
#         self.labels = labels
    
#     def prepare_data(self) -> None:
#         AudioDataset(
#             destination=self.destination, 
#             set_type=None, 
#             audio_rate=self.audio_rate, 
#             labels=self.labels
#         )
    
#     def setup(self, stage: str) -> None:
#         if stage == "fit" or stage is None:
#             self.train_set = AudioDataset(self.destination, "training", self.audio_rate, self.labels)
#             self.val_set = AudioDataset(self.destination, "validation", self.audio_rate, self.labels)
#         if stage == "test" or stage is None:
#             self.test_set = AudioDataset(self.destination, "testing", self.audio_rate, self.labels)
    
#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         return DataLoader(
#             dataset=self.train_set,
#             shuffle=True,
#             batch_size=self.batch_size
#         )
    
#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             dataset=self.val_set,
#             shuffle=False,
#             batch_size=self.batch_size
#         )
    
#     def test_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             dataset=self.test_set,
#             shuffle=False,
#             batch_size=self.batch_size
#         )

def preprocess_waveform(waveform, length=16000, transform=None, padLR=False):
    padding = int((length - waveform.shape[1]))
    if padLR:
        waveform = torch.roll(torch.nn.functional.pad(waveform, (0, padding)), padding // 2)
    else:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    if transform is not None:
        features = transform(waveform)
    return(features)


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, path = '.', transform = None, subset:Literal[None, "training", "validation", "testing"] = None, debug_size=-1):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            path,
            url='speech_commands_v0.02',
            folder_in_archive='SpeechCommands',
            download=True,
            subset=subset
        )
        self.transform = transform

        # unknown word results in a default value of len(KNOWN_COMMANDS)
        self.word2num = defaultdict(lambda: len(KNOWN_COMMANDS)-1)
        for num, command in enumerate(KNOWN_COMMANDS):
            self.word2num[command] = num

        if debug_size > 0:
            self.dataset = Subset(self.dataset, torch.arange(debug_size))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        (waveform, sample_rate, label, _, _) = self.dataset[index]
        features = preprocess_waveform(waveform, sample_rate, self.transform)
        label = self.word2num[label]

        return features, label


class AudioDataModule(L.LightningDataModule):
    def __init__(self, data_destination, batch_size, n_mels, hop_length, debug_size=-1) -> None:
        super().__init__()

        self.data_destination = data_destination
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.debug_size = debug_size

        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )

        self.train_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_mels*0.2)),
            torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000/160)),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
    
    def prepare_data(self) -> None:
        # self.sc_data = SpeechCommandsData(
        #     path=self.data_destination, 
        #     train_bs=self.batch_size, 
        #     test_bs=self.batch_size, 
        #     val_bs=self.batch_size, 
        #     n_mels=self.n_mels,
        #     hop_length=self.hop_length,
        #     debug_size=self.debug_size
        # )

        print('Initialize/download SpeechCommandsDataset...')
        self.train_dataset = SpeechCommandsDataset(path=self.data_destination, transform=self.train_transform, subset="training", debug_size=self.debug_size)
        self.val_dataset = SpeechCommandsDataset(path=self.data_destination, transform=self.transform, subset="validation", debug_size=self.debug_size)
        self.test_dataset = SpeechCommandsDataset(path=self.data_destination, transform=self.transform, subset="testing", debug_size=self.debug_size)
        print("Dataset is initialize/download now")
        
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        if stage == "test" or stage is None:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader