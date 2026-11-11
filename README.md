# VoiceCat

<img src="desktop/icons/icon.png" alt="VoiceCat" width="128">

把本机正在播放的声音转成文字。听的是系统输出，不采集麦克风。桌面窗口只在本机打开。

## 能做什么

- 边听边出字。流式 Zipformer 会随着声音往下写；Whisper、faster-whisper、SenseVoice、Paraformer 按短段识别。
- 识别页左右对照。左边是原文，右边是润色。一句结束后原文固定，只有正在听的那一句会继续变。
- 大模型润色。给原文补标点，去掉重复，改明显错词。可以改润色提示词，也可以维护专有名词。
- 记住正确写法。在原文里选中错词，填上要替换的说法，这次和下次都会换。
- 语言可选简体中文、繁体中文、英文，或交给模型自动检测。
- 识别引擎可选 Whisper、faster-whisper，以及 sherpa 的 SenseVoice、Paraformer、流式 Zipformer。已下载的模型会出现在模型页。
- 默认识别在本机完成。也可以把润色或在线转写指到自己的接口。
- 左边栏可以收起。设置单独打开，里面有音频输入、模型、下载和提示词。

## 支持的系统

| 系统 | 现状 |
| --- | --- |
| macOS 15 及以上 | 可以打包桌面窗口，并用系统声音采集。采集程序按 macOS 15 编译。 |
| Windows | 可以用 WASAPI 环回听系统声音，走命令行。需要安装 `pyaudiowpatch`。还没有桌面安装包。 |
| Linux | 还没有系统声音采集，暂不能用。 |

## 运行

桌面版目前只在 macOS 上打包。需要 Python 3.12，以及编译好的采集程序 `bin/sysaudio-capture`。Intel Mac 上 PyTorch 不要超过 2.2，`transformers` 固定 4.46.3。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./build.sh
./run.sh --list
./package_mac.sh
```

打包后从 `~/Applications/VoiceCat.app` 打开。命令行仍可用 `./run.sh`。

Windows 上先安装 `pyaudiowpatch`，再用 `./run.sh` 听系统输出。

## 开发

```bash
.venv/bin/python -m unittest discover -s tests -v
cd desktop && cargo build
```

界面在 `sysaudio_stt/web.py`。桌面壳在 `desktop/`。

## 许可

MIT。见 `LICENSE`。
