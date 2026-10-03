import pickle
import sounddevice
import numpy
from scipy.signal import butter, sosfilt, sosfilt_zi

# initialising libraries
_7_channel_standard_cord_dic = {6: 15, 8: 355, 7: 0, 17: 0, 11: 0, 15: 0, 13: 0}
song_samplerate = 48000

# mixer board(for dump people, you can adjust each channel volume here)
FL_vol = 1  # aka left in stereo defaults 1
FR_vol = 1  # aka right in stereo defaults 1
C_vol = 1  # default 1.5
SL_vol = 0.9
SR_vol = 0.9
BL_vol = 0.9
BR_vol = 0.9
SIR_vol = 0.2
SIL_vol = 0.2
bass_vol = 0.3
wet_mix_vol = 0.002
master_vol = 1.7

with open(r"C:\Spatial Engine\spatial_engine_libraries.pkl", "rb") as file:
    master_library = pickle.load(file)
fast_look_up_7c = master_library["7.1"]
rir = master_library["reverb"]
speaker_states = master_library["speaker_properties"]

lp_sos = butter(4, 120, "lowpass", fs=song_samplerate, output="sos")
left_lp_zi = sosfilt_zi(lp_sos) * 0.0
right_lp_zi = sosfilt_zi(lp_sos) * 0.0
hp_sos = butter(4, 120, "highpass", fs=song_samplerate, output="sos")
left_hp_zi = sosfilt_zi(hp_sos) * 0.0
right_hp_zi = sosfilt_zi(hp_sos) * 0.0


total_rir_chunk = 46

left_leftover_chunk_cache = numpy.zeros(1024, dtype=numpy.float32)
right_leftover_chunk_cache = numpy.zeros(1024, dtype=numpy.float32)
audio_history = numpy.zeros((total_rir_chunk, 1025), dtype=numpy.complex128)
print("engine parameter build done")

is_turning = False


# reverb engine to generate wet mix


def reverb_engine(
    mono_chunk,
    left_half_hrtf_vector,
    right_half_hrtf_vector,
    frames,
    audio_history,
    left_leftover_chunk_cache,
    right_leftover_chunk_cache,
):
    mono_chunk = numpy.fft.rfft(numpy.pad(mono_chunk, (0, 1024)))
    audio_history[1:] = audio_history[:-1]
    audio_history[0] = mono_chunk
    new_left_chunk = audio_history * left_half_hrtf_vector
    new_left_spatial_chunk = numpy.sum(new_left_chunk, axis=0)
    spatial_left = numpy.fft.irfft(new_left_spatial_chunk)
    spatial_left[:frames] += left_leftover_chunk_cache
    left_leftover_chunk_cache[:] = spatial_left[frames:]
    final_left_spatial = spatial_left[:frames]
    new_right_chunk = audio_history * right_half_hrtf_vector
    new_right_spatial_chunk = numpy.sum(new_right_chunk, axis=0)
    spatial_right = numpy.fft.irfft(new_right_spatial_chunk)
    spatial_right[:frames] += right_leftover_chunk_cache
    right_leftover_chunk_cache[:] = spatial_right[frames:]
    final_right_spatial = spatial_right[:frames]
    return numpy.column_stack((final_left_spatial, final_right_spatial))


# spatial audio engine takes inn incoming audio chunk and using hrtf files gives it positional effects
def spatial_engine(mono_chunk, left_hrtf, right_hrtf, frames, state):
    global is_turning
    mono_chunk = numpy.fft.rfft(numpy.pad(mono_chunk, (0, 3072)))
    left_hrtf = left_hrtf
    right_hrtf = right_hrtf

    if is_turning == False:
        new_L = mono_chunk * left_hrtf
        new_R = mono_chunk * right_hrtf
        new_L = numpy.fft.irfft(new_L)
        new_R = numpy.fft.irfft(new_R)

        new_L[: len(state["L_tail"])] += state["L_tail"]
        new_R[: len(state["R_tail"])] += state["R_tail"]

        state["L_tail"][:] = new_L[frames : frames + 1999]
        state["R_tail"][:] = new_R[frames : frames + 1999]

        state["old_L_hrtf"] = left_hrtf
        state["old_R_hrtf"] = right_hrtf

        return numpy.column_stack((new_L[:frames], new_R[:frames]))

    else:
        new_L = mono_chunk * left_hrtf
        new_R = mono_chunk * right_hrtf
        new_L = numpy.fft.irfft(new_L)
        new_R = numpy.fft.irfft(new_R)

        old_L = mono_chunk * state["old_L_hrtf"]
        old_R = mono_chunk * state["old_R_hrtf"]
        old_L = numpy.fft.irfft(old_L)
        old_R = numpy.fft.irfft(old_R)

        new_full = numpy.column_stack((new_L[:frames], new_R[:frames]))
        old_full = numpy.column_stack((old_L[:frames], old_R[:frames]))

        faded_in = fade_in(new_full, frames)
        faded_out = fade_out(old_full, frames)

        faded_in[:, 0][: len(state["L_tail"])] += state["L_tail"]
        faded_in[:, 1][: len(state["R_tail"])] += state["R_tail"]

        state["L_tail"][:] = new_L[frames : frames + 1999]
        state["R_tail"][:] = new_R[frames : frames + 1999]

        state["old_L_hrtf"] = left_hrtf
        state["old_R_hrtf"] = right_hrtf
        is_turning = False

        return numpy.clip((faded_in + faded_out), -1.0, 1.0)


current_angle = 0
mono_chunk = 0
previous_current_dist = 1.0
stereo_flag = False
multi_flag = False
total_timer = 200
current_timer = 0


def fade_in(new_spatial_chunk, frames):
    return new_spatial_chunk * numpy.linspace(0.0, 1.0, frames)[:, numpy.newaxis]


def fade_out(old_spatial_chunk, frames):
    return old_spatial_chunk * numpy.linspace(1.0, 0.0, frames)[:, numpy.newaxis]


def channel_finder_and_assigner(indata, frames, outdata):
    global is_turning
    global mono_chunk
    global lp_sos
    global hp_sos
    global left_lp_zi
    global right_lp_zi
    global left_hp_zi
    global right_hp_zi
    global speaker_states
    global stereo_flag
    global multi_flag
    global current_timer
    global total_timer
    multi_check = indata[:, 2:]
    stereo = indata[:, :2]
    master_stereo_chunk = numpy.zeros((frames, 2), dtype=numpy.float32)

    speaker = list(_7_channel_standard_cord_dic.keys())
    FL_left_half_hrtf_vector = fast_look_up_7c[(speaker[0], current_angle)][0]
    FL_right_half_hrtf_vector = fast_look_up_7c[(speaker[0], current_angle)][1]
    FR_left_half_hrtf_vector = fast_look_up_7c[(speaker[1], current_angle)][0]
    FR_right_half_hrtf_vector = fast_look_up_7c[(speaker[1], current_angle)][1]
    C_left_half_hrtf_vector = fast_look_up_7c[(speaker[2], current_angle)][0]
    C_right_half_hrtf_vector = fast_look_up_7c[(speaker[2], current_angle)][1]
    SL_left_half_hrtf_vector = fast_look_up_7c[(speaker[3], current_angle)][0]
    SL_right_half_hrtf_vector = fast_look_up_7c[(speaker[3], current_angle)][1]
    SR_left_half_hrtf_vector = fast_look_up_7c[(speaker[4], current_angle)][0]
    SR_right_half_hrtf_vector = fast_look_up_7c[(speaker[4], current_angle)][1]
    BL_left_half_hrtf_vector = fast_look_up_7c[(speaker[5], current_angle)][0]
    BL_right_half_hrtf_vector = fast_look_up_7c[(speaker[5], current_angle)][1]
    BR_left_half_hrtf_vector = fast_look_up_7c[(speaker[6], current_angle)][0]
    BR_right_half_hrtf_vector = fast_look_up_7c[(speaker[6], current_angle)][1]
    #     SIL_left_half_hrtf_vector = fast_look_up_7c[(speaker[7], current_angle)][0]
    #     SIL_right_half_hrtf_vector = fast_look_up_7c[(speaker[7], current_angle)][1]
    #     SIR_left_half_hrtf_vector = fast_look_up_7c[(speaker[8], current_angle)][0]
    #     SIR_right_half_hrtf_vector = fast_look_up_7c[(speaker[8], current_angle)][1]
    raw_chunk = indata

    multi_channel = raw_chunk[:, 2:]
    if numpy.abs(numpy.max(multi_channel)) > 0.001:
        current_timer = total_timer
        Front_left_mono_chunk = (raw_chunk[:, 0]) / 2.0
        Front_right_mono_chunk = (raw_chunk[:, 1]) / 2.0
        Centre_mono_chunk = (raw_chunk[:, 2]) / 2.0
        lowpass_mono_chunk = (raw_chunk[:, 3]) / 2.0
        Surround_left_mono_chunk = (raw_chunk[:, 4]) / 2.0
        Surround_right_mono_chunk = (raw_chunk[:, 5]) / 2.0
        Back_left_mono_chunk = (raw_chunk[:, 6]) / 2.0
        Back_right_mono_chunk = (raw_chunk[:, 7]) / 2.0

    else:
        if current_timer > 0:
            current_timer -= 1
            Front_left_mono_chunk = (raw_chunk[:, 0]) / 2.0
            Front_right_mono_chunk = (raw_chunk[:, 1]) / 2.0
            Centre_mono_chunk = (raw_chunk[:, 2]) / 2.0
            lowpass_mono_chunk = (raw_chunk[:, 3]) / 2.0
            Surround_left_mono_chunk = (raw_chunk[:, 4]) / 2.0
            Surround_right_mono_chunk = (raw_chunk[:, 5]) / 2.0
            Back_left_mono_chunk = (raw_chunk[:, 6]) / 2.0
            Back_right_mono_chunk = (raw_chunk[:, 7]) / 2.0
    
        else:
            left_mono_chunk_unfiltered = raw_chunk[:, 0]
            left_lowpass, left_lp_zi = sosfilt(
                sos=lp_sos, zi=left_lp_zi, x=left_mono_chunk_unfiltered
            )
            left_mono_chunk, left_hp_zi = sosfilt(
                sos=hp_sos, zi=left_hp_zi, x=left_mono_chunk_unfiltered
            )
            right_mono_chunk_unfiltered = raw_chunk[:, 1]
            right_lowpass, right_lp_zi = sosfilt(
                sos=lp_sos, zi=right_lp_zi, x=right_mono_chunk_unfiltered
            )
            right_mono_chunk, right_hp_zi = sosfilt(
                sos=hp_sos, zi=right_hp_zi, x=right_mono_chunk_unfiltered
            )

            lowpass_mono_chunk = (left_lowpass + right_lowpass) * 0
            Centre_mono_chunk = (
                left_mono_chunk_unfiltered + right_mono_chunk_unfiltered
            ) / 2.0
            Surround_left_mono_chunk = (left_mono_chunk - right_mono_chunk) / 2.0
            Surround_right_mono_chunk = (right_mono_chunk - left_mono_chunk) / 2.0
            Front_left_mono_chunk = left_mono_chunk_unfiltered
            Front_right_mono_chunk = right_mono_chunk_unfiltered
            Back_left_mono_chunk = Surround_left_mono_chunk / 2.0
            Back_right_mono_chunk = Surround_right_mono_chunk / 2.0
       

    Front_left_spatial = (
        spatial_engine(
            Front_left_mono_chunk,
            FL_left_half_hrtf_vector,
            FL_right_half_hrtf_vector,
            frames,
            speaker_states["FL"],
        )
    ) * FL_vol
    Front_right_spatial = (
        spatial_engine(
            Front_right_mono_chunk,
            FR_left_half_hrtf_vector,
            FR_right_half_hrtf_vector,
            frames,
            speaker_states["FR"],
        )
    ) * FR_vol
    Centre_spatial = (
        spatial_engine(
            Centre_mono_chunk,
            C_left_half_hrtf_vector,
            C_right_half_hrtf_vector,
            frames,
            speaker_states["C"],
        )
    ) * C_vol
    Surround_left_spatial = (
        spatial_engine(
            Surround_left_mono_chunk,
            SL_left_half_hrtf_vector,
            SL_right_half_hrtf_vector,
            frames,
            speaker_states["SL"],
        )
    ) * SL_vol
    Surround_right_spatial = (
        spatial_engine(
            Surround_right_mono_chunk,
            SR_left_half_hrtf_vector,
            SR_right_half_hrtf_vector,
            frames,
            speaker_states["SR"],
        )
    ) * SR_vol
    Back_left_spatial = (
        spatial_engine(
            Back_left_mono_chunk,
            BL_left_half_hrtf_vector,
            BL_right_half_hrtf_vector,
            frames,
            speaker_states["BL"],
        )
    ) * BL_vol
    Back_right_spatial = (
        spatial_engine(
            Back_right_mono_chunk,
            BR_left_half_hrtf_vector,
            BR_right_half_hrtf_vector,
            frames,
            speaker_states["BR"],
        )
    ) * BR_vol
    #     Surround_wide_left_spatial = (
    #         spatial_engine(
    #             Surround_left_mono_chunk,
    #             SIL_left_half_hrtf_vector,
    #             SIL_right_half_hrtf_vector,
    #             frames,
    #             speaker_states["SIL"],
    #         )
    #     ) * SIL_vol
    #     Surround_wide_right_spatial = (
    #         spatial_engine(
    #             Surround_right_mono_chunk,
    #             SIR_left_half_hrtf_vector,
    #             SIR_right_half_hrtf_vector,
    #             frames,
    #             speaker_states["SIR"],
    #         )
    #     ) * SIR_vol

    # adding reverb
    center_mono_mix = (
        Front_left_mono_chunk
        + Front_right_mono_chunk
        + Centre_mono_chunk
        + Surround_left_mono_chunk
        + Surround_right_mono_chunk
    ) / 2.0
    room_rev = (
        reverb_engine(
            center_mono_mix,
            rir["left_rir"],
            rir["right_rir"],
            frames,
            audio_history,
            left_leftover_chunk_cache,
            right_leftover_chunk_cache,
        )
    ) * wet_mix_vol
    # final mix aaray
    master_stereo_chunk += Front_left_spatial
    master_stereo_chunk += Front_right_spatial
    master_stereo_chunk += Centre_spatial
    master_stereo_chunk += Surround_left_spatial
    master_stereo_chunk += Surround_right_spatial
    #     master_stereo_chunk += Surround_wide_left_spatial
    #     master_stereo_chunk += Surround_wide_right_spatial
    master_stereo_chunk += Back_left_spatial
    master_stereo_chunk += Back_right_spatial
    # reverb
    master_stereo_chunk += room_rev
    # bass
    master_stereo_chunk[:, 0] += (lowpass_mono_chunk) * bass_vol
    master_stereo_chunk[:, 1] += (lowpass_mono_chunk) * bass_vol
    return master_stereo_chunk * master_vol


def auto_callback(indata, outdata, frames, time, status):
    outdata[:] = channel_finder_and_assigner(indata, frames, outdata)


# keyboard.add_hotkey('right',pan_right)
# keyboard.add_hotkey('left',pan_left)

wasapi_idx = next(
    i for i, host in enumerate(sounddevice.query_hostapis()) if "WASAPI" in host["name"]
)


in_id = next(
    i
    for i, dev in enumerate(sounddevice.query_devices())
    if "CABLE Output" in dev["name"] and dev["hostapi"] == wasapi_idx
)
out_id = next(
    i
    for i, dev in enumerate(sounddevice.query_devices())
    if "Headphones" in dev["name"] and dev["hostapi"] == wasapi_idx
)

stream = sounddevice.Stream(
    device=(in_id, out_id),
    samplerate=48000,
    channels=(8, 2),
    blocksize=1024,
    callback=auto_callback,
)

with stream:
    input("Live Engine Routing Active... Press Enter to stop.\n")
