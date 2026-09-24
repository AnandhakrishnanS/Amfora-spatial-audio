🎧 Amphora Engine

Amphora Engine is a real-time, low-latency 7.1/9.1 spatial audio DSP engine written in Python. It intercepts multi-channel audio via WASAPI, applies Head-Related Transfer Functions (HRTFs), and outputs binaural or upmixed stereo directly to headphones.
✨ Core Features

    ⏱️ Real-Time HRTF Convolution: Uses Fast Fourier Transforms (rfft/irfft) to apply complex spatial acoustic vectors without CPU bottlenecks.

    🌊 Partitioned Reverb Engine: Slices room acoustic tails into discrete 1024-sample arrays for highly efficient delay line processing.

    🧩 Overlap-Add Tail Management: Mathematically catches and loops excess FFT convolution tails into the next frame to completely eliminate audio popping.

    🤖 Auto-Switching State Machine: Uses a 4.2-second (200-frame) countdown timer to intelligently switch between 7.1 spatialization and stereo upmixing without ruining intentional cinematic silence.

    🎛️ Stateful Crossovers: Implements 4th-order Butterworth filters (sosfilt_zi) that successfully pass state across chunk boundaries to prevent phase smearing.

🏗️ Architecture

Built on a zero-allocation I/O callback loop processing 1024-sample chunks every ~21.3ms. It uses dynamic pathing to calculate its own physical memory location, entirely preventing CWD shortcut crashes when compiled as a standalone binary:
Python

import sys, os

if getattr(sys, 'frozen', False):
    dir_path = os.path.dirname(sys.executable)
else:
    dir_path = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(dir_path, "spatial_engine_libraries.pkl")

🛠️ Installation & Build

Prerequisites: Python 3.12+, numpy, scipy, sounddevice, pyinstaller.
(Note: Requires a pre-compiled spatial_engine_libraries.pkl containing the HRTF arrays to operate).

Compile Standalone .exe:
Bash

python -m PyInstaller --clean --noconsole --onedir spatial_engine_v2.py

Deployment:

1.intall the vb cable drive using the above setup wizard located in the "install this before running" folder.

2.Restart you pc

3.search for sound in the windows start search bar

4. click on playback tab

5.look for "CABLE input" not the 16ch one(if it appears), click on it(single click) 

6.under there will be a configure button ,from the list select 7.1 ,hit next and at last press finish

7.put "CABLEinput" as your output device

7.your done,now you can use amfora,just double click on the EXE file

***notice***
1.curently amfora is in alpha face so in order to quit the program you need to kill it from task manger currently.

2.Amfora only works if an headphone is connected to the computer ,as it is build around a headphone and not for speakers(its not a bug if it throws an error while no head phones are in)

3.don't forget to switch back to your default listening device after using the software.






