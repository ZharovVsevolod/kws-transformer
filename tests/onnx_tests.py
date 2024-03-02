import onnxruntime as ort
import numpy as np
import torch
from kws_transformer.kws.preprocessing_data.preproccesing import SpeechCommandsData

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

save_onnx = "final_weights/model.onnx"
input_sample = torch.randn((1, 1, 40, 100))

ort_session = ort.InferenceSession(save_onnx)

input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: input_sample.numpy()}

ort_outs = ort_session.run(None, ort_inputs)

tensor_outputs = torch.tensor(ort_outs)[:, :,-1,:].squeeze()
print(tensor_outputs.shape)
print(tensor_outputs)
answer_class = torch.argmax(tensor_outputs)
print(answer_class, KNOWN_COMMANDS[answer_class])


sc_data = SpeechCommandsData(
    path="./kws_transformer/dataset",
    train_bs=512,
    test_bs=512, 
    val_bs=512, 
    n_mels=40,
    hop_length=161
)

print(len(sc_data.train_dataset))
print(len(sc_data.val_dataset))
print(len(sc_data.test_dataset))

x, y = sc_data.train_dataset[13]
print(x.shape)
print(y)