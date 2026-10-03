import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt, sosfilt_zi
import numpy
from pathlib import Path
from math import sqrt, pow, dist, ceil
import sofa
import pickle

_7_channel_standard_cord_dic = {6: 15, 8: 355, 7: 0, 17: 0, 11: 0, 15: 15, 13: 355}

hrtf_path = Path(r"D:\adm downloads\Compressed\bbcrdlr_all_speakers.sofa")
hrtf = sofa.Database.open(hrtf_path)
angle_index_dic = {}
listener_data = hrtf.Listener.View.get_values(system="spherical", angle_unit="degree")
all_ir_data = hrtf.Data.IR.get_values()

for index, angles in enumerate(listener_data):
    if angles[0] < 0:
        angle_index_dic[index] = angles[0] + 360
    else:
        angle_index_dic[index] = angles[0]


# fast lookup_table for a speaker containing its column index and and angles  as key and in values ,right hrtf and ;eft hrtf data
fast_look_up_7c = {}

# finding index for physical speaker for 7.1
for speaker in _7_channel_standard_cord_dic.keys():
    base_angle = _7_channel_standard_cord_dic[speaker]
    for angle in range(0, 360, 2):
        actual_angle = (base_angle - angle) % 360
        closest_index = min(
            angle_index_dic.keys(),
            key=lambda ids: abs(angle_index_dic[ids] - actual_angle),
        )
        hrtf_files = all_ir_data[closest_index]
        left_hrtf = hrtf_files[0][speaker][:2000]
        right_hrtf = hrtf_files[1][speaker][:2000]
        left_hrtf[1900:] = left_hrtf[1900:] * numpy.linspace(
            1.0, 0.0, len(left_hrtf[1900:])
        )
        right_hrtf[1900:] = right_hrtf[1900:] * numpy.linspace(
            1.0, 0.0, len(right_hrtf[1900:])
        )
        left_hrtf = numpy.fft.rfft(numpy.pad(left_hrtf, (0, 4096 - len(left_hrtf))))
        right_hrtf = numpy.fft.rfft(numpy.pad(right_hrtf, (0, 4096 - len(right_hrtf))))
        fast_look_up_7c[(speaker, angle)] = (left_hrtf, right_hrtf)


speaker_states = {
    "FL": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "FR": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "SL": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "SR": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "C": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "BL": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "BR": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "SIL": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
    "SIR": {
        "L_tail": numpy.zeros(1999, dtype=numpy.float32),
        "R_tail": numpy.zeros(1999, dtype=numpy.float32),
        "old_L_hrtf": 0,
        "old_R_hrtf": 0,
    },
}

# reverb library build
reverb_vector, reverb_samplerate = sf.read(
    r"D:\adm downloads\Compressed\StereoBinauralIRPos2D21Cin.wav"
)
left_rir = reverb_vector[500:, 0]
right_rir = reverb_vector[500:, 1]
rir = {"left_rir": [], "right_rir": []}
curent_iteration = 1024
init_iter = 0
total_rir_chunk = ceil(len(left_rir) / 1024)
for sample in range(0, total_rir_chunk):
    left_rir_siced = left_rir[init_iter:curent_iteration]
    right_rir_siced = right_rir[init_iter:curent_iteration]
    left_rir_siced = numpy.fft.rfft(
        numpy.pad(left_rir_siced, (0, 2048 - len(left_rir_siced)))
    )
    right_rir_siced = numpy.fft.rfft(
        numpy.pad(right_rir_siced, (0, 2048 - len(right_rir_siced)))
    )
    rir["left_rir"].append(left_rir_siced)
    rir["right_rir"].append(right_rir_siced)
    init_iter = curent_iteration
    curent_iteration += 1024
rir["left_rir"] = numpy.array(rir["left_rir"])
rir["right_rir"] = numpy.array(rir["right_rir"])


master_lib = {
    "7.1": fast_look_up_7c,
    "reverb": rir,
    "speaker_properties": speaker_states,
}

# saving to ssd
with open(r"C:\Spatial Engine\spatial_engine_libraries.pkl", "wb") as file:
    pickle.dump(master_lib, file)
