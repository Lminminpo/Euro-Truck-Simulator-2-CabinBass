# Euro-Truck-Simulator-2-CabinBass
一个给音乐加驾驶室混响的工具    A tool that adds cabin reverb effects to music for truck simulator players

|

什么原理    How it works

音乐文件 → 实时音频处理（模拟驾驶室混响） → 耳机    Music file → Real-time audio processing (simulating cabin reverb) → Headphones

|

用法和功能    Usage and Features

下载exe文件，双击打开，选择你存放音乐的文件夹，F9用来播放和暂停，wasd四个键分别用来关闭右车窗，打开左车窗，关闭左车窗，打开右车窗（此功能不会影响游戏内的车窗状态，建议将游戏内控制车窗的按键设置成和工具对应以获得最好的体验）以模拟不同状态下的声音效果    Download the exe file, double-click to open, select your music folder. F9 to play/pause. A/S/W/D keys to open left window, close left window, close right window, open right window respectively (this tool does not affect the in-game window state, it is recommended to set the in-game window control keys to match the tool for the best experience), simulating sound effects under different window states

|

怎么卸载    How to uninstall

删除exe文件和同一文件夹内的cfg文件即可    Just delete the exe file and the cfg file in the same folder

|

从源码运行    Run from source

pip install numpy scipy sounddevice pynput miniaudio

python test_music.py

打包    Build

pip install pyinstaller

pyinstaller --onefile --noconsole --name CabinBass test_music.py

|

下载    Download

https://wwayt.lanzoul.com/iirnQ3pxyq5c
密码:hpbt
