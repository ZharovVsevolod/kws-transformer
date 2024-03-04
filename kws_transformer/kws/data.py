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
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

from pathlib import Path
from typing import Optional, Tuple, Union
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
from torch import Tensor


FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SAMPLE_RATE = 16000
_CHECKSUMS = {
    "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}

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

def balance_classes(whole_dataset, size_of_class, word2num, more_background = 1):
    cut_dataset = []
    full_size = np.full(11, size_of_class)
    full_size[10] = full_size[10] * more_background

    global_temp = 0
    print(f"Нормирование классов, всех классов будет {size_of_class}, всего по классам - {full_size}")

    for (waveform, sample_rate, label, t, r) in whole_dataset:
        label_num = word2num[label]
        if full_size[label_num] != 0:
            cut_dataset.append((waveform, sample_rate, label, t, r))
            full_size[label_num] -= 1
        temp = 0
        for s in full_size:
            if s == 0:
                temp += 1
        
        if temp == 11:
            break
        else:
            if temp > global_temp:
                global_temp = temp
                print(f"Заполнение классов закончено на {round(global_temp / 11, 2)}%, осталось по классам: {full_size}")
    

    print(f"Все классы заполнены (или кончился датасет), предполагаемое равное количество - {size_of_class} (или сколько было, не набрано по классам - {full_size})")
    return cut_dataset
        

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            path = '.', 
            transform = None, 
            subset:Literal[None, "training", "validation", "testing"] = None, 
            debug_size=-1,
            more_background = 1,
            train_list = False
        ):
        self.dataset = SPEECHCOMMANDS(
            path,
            url='speech_commands_v0.02',
            folder_in_archive='SpeechCommands',
            download=True,
            subset=subset,
            train_list=train_list
        )
        self.transform = transform

        # unknown word results in a default value of len(KNOWN_COMMANDS)
        self.word2num = defaultdict(lambda: len(KNOWN_COMMANDS)-1)
        for num, command in enumerate(KNOWN_COMMANDS):
            self.word2num[command] = num

        # if debug_size > 0:
            # self.dataset = balance_classes(self.dataset, debug_size, self.word2num, more_background)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        (waveform, sample_rate, label, _, _) = self.dataset[index]
        features = preprocess_waveform(waveform, sample_rate, self.transform)
        label = self.word2num[label]

        return features, label


class AudioDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_destination, 
            batch_size, 
            n_mels, 
            hop_length, 
            debug_size=-1, 
            more_background=1, 
            background=True
        ) -> None:
        super().__init__()

        self.data_destination = data_destination
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.debug_size = debug_size
        self.more_background = more_background
        self.background = background
        if self.background and self.debug_size > 0:
            self.train_list = True
        else:
            self.train_list = False

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
        self.train_dataset = SpeechCommandsDataset(
            path=self.data_destination, 
            transform=self.train_transform, 
            subset="training", 
            debug_size=self.debug_size, 
            more_background=self.more_background,
            train_list=self.train_list
        )
        self.val_dataset = SpeechCommandsDataset(
            path=self.data_destination, 
            transform=self.transform, 
            subset="validation", 
            debug_size=self.debug_size, 
            more_background=self.more_background
        )
        self.test_dataset = SpeechCommandsDataset(
            path=self.data_destination, 
            transform=self.transform, 
            subset="testing", 
            debug_size=self.debug_size, 
            more_background=self.more_background
        )
        print("Dataset is initialize/download now")

        #-----
        
        if self.background:
            # Cleanup background files if needed
            backgroundDir = os.path.join(self.data_destination, 'SpeechCommands', 'speech_commands_v0.02', 'background')
            if (os.path.isdir(backgroundDir)):
                print('Found existing background directory. Removing files in that directory.')
                shutil.rmtree(backgroundDir)

            # Generate background files: creates files with 1 sec duration
            self.background_fileList = generate_background_files(self.data_destination + "/SpeechCommands/speech_commands_v0.02/_background_noise_")
            print(f'Background files generated: {len(self.background_fileList)}\n')

            self.val_ratio = len(self.val_dataset) / len(self.train_dataset)
            self.test_ratio = len(self.test_dataset) / len(self.train_dataset)
            self.train_ratio = 1.0 - self.val_ratio - self.test_ratio

            # if True:
            #     temp = self.train_dataset.dataset._walker
            #     print("Общая выборка: ", len(temp))
            #     self.train_dataset.dataset._walker, temp_valtest = train_test_split(temp, test_size=0.2)
            #     del temp

            #     print("Тренировочные данные: ", len(self.train_dataset.dataset._walker))
            #     self.val_dataset.dataset._walker, self.test_dataset.dataset._walker = train_test_split(temp_valtest, test_size=0.5)
            #     del temp_valtest
            
            # Add background files to walker
            idx_train = int(self.train_ratio * len(self.background_fileList))
            self.train_dataset.dataset._walker += self.background_fileList[:idx_train]
            idx_val = idx_train + int(self.val_ratio * len(self.background_fileList))
            self.val_dataset.dataset._walker += self.background_fileList[idx_train:idx_val]
            idx_test = idx_val + int(self.test_ratio * len(self.background_fileList))
            self.test_dataset.dataset._walker += self.background_fileList[idx_val:idx_test]

        #----
        
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

def generate_background_files(dataset_path, num_samples=16000):

    background_source_files = glob.glob(os.path.join(
        dataset_path, "_background_noise_", "*.wav"))

    targetDir = os.path.join(dataset_path, "background")
    # Generate Background Files
    print('Generate 1s background files:\n')
    os.makedirs(targetDir, exist_ok=True)
    for f in background_source_files:
        waveform, sr = torchaudio.load(f)
        split_waveforms = torch.split(waveform, num_samples, dim=1)
        for idx, split_waveform in enumerate(split_waveforms):
            torchaudio.save(os.path.join(
                targetDir, f'{hash(waveform)}_nohash_{idx}.wav'), split_waveform, sample_rate=sr)

    background_target_files = glob.glob(
        os.path.join(targetDir, "*.wav"))
    return(background_target_files)

#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

class SPEECHCOMMANDS(Dataset):
    """*Speech Commands* :cite:`speechcommandsv2` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        train_list = False
    ) -> None:

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "http://download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive, self._path)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            if train_list:
                self._walker = _load_list(self._path, "training_list.txt")
            else:
                excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
                walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
                self._walker = [
                    w
                    for w in walker
                    if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
                ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to the audio
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)



def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def _get_speechcommands_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, SAMPLE_RATE, label, speaker_id, utterance_number