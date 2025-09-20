# 🎬 Auto Subtitle & Translate with FunASR / Whisper + LLaMA / Google Translate

🌐 **Language**: [English](README.md) | [繁體中文](README_zh-TW.md)

Automatically generate subtitles for any video using
[FunASR](https://github.com/alibaba-damo-academy/FunASR) or
[OpenAI Whisper](https://openai.com/blog/whisper), overlay them with `ffmpeg`,
and optionally translate them to 50+ languages using a
[LLaMA2-based multilingual translator](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate)
or [Googletrans](https://github.com/ssut/py-googletrans).

## 📺 [Demo Video](https://youtu.be/vkvTpmQ7M48?si=qQLvYzwtsQ4djo4K)

![Auto Subtitle Demo Screenshot 1](https://github.com/YJ-20/auto-subtitle-translate/assets/68987494/85a41810-75ac-44f8-9b75-35c599032619)

![Auto Subtitle Demo Screenshot 2](https://github.com/YJ-20/auto-subtitle-translate/assets/68987494/88d42ad7-da9f-4749-9923-4ec9fc9ed040)

![Auto Subtitle Demo Screenshot 3](https://github.com/YJ-20/auto-subtitle-translate/assets/68987494/1c255fae-a1c5-4cb1-a60c-87a6aabfcf04)

![Auto Subtitle Demo Screenshot 4](https://github.com/YJ-20/auto-subtitle-translate/assets/68987494/91ad2860-18a7-460c-91e6-011265308433)

---

## 🛠️ Installation

Make sure you have Python 3.7 or later.

Install the package directly from GitHub:

```bash
pip install git+https://github.com/e2720pjk/auto-subtitle-translate
```

**Additional dependencies for FunASR and Google Translate:**

```bash
pip install funasr librosa torch torchaudio numpy scipy soundfile googletrans
```

### Install `ffmpeg`

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

---

## 🚀 Usage

### Only Transcribe (w/o Translate)

```bash
auto_subtitle_cli /path/to/video.mp4
```

### Transcribe and Translate Subtitles

To translate subtitles to another language:

```bash
# Using LLaMA2 backend (default)
auto_subtitle_cli /path/to/video.mp4 --translate_to ko_KR

# Using Google Translate backend
# (recommended for Traditional Chinese and better accuracy)
auto_subtitle_cli /path/to/video.mp4 --translate_to zh_TW \
  --translator_backend googletrans
```

---

## 🌐 Supported Translation Languages

| Language     | Code   | Language     | Code   | Language     | Code   |
|--------------|--------|--------------|--------|--------------|--------|
| Arabic       | ar_AR  | Czech        | cs_CZ  | German       | de_DE  |
| English      | en_XX  | Spanish      | es_XX  | Estonian     | et_EE  |
| Finnish      | fi_FI  | French       | fr_XX  | Gujarati     | gu_IN  |
| Hindi        | hi_IN  | Italian      | it_IT  | Japanese     | ja_XX  |
| Kazakh       | kk_KZ  | Korean       | ko_KR  | Lithuanian   | lt_LT  |
| Latvian      | lv_LV  | Burmese      | my_MM  | Nepali       | ne_NP  |
| Dutch        | nl_XX  | Romanian     | ro_RO  | Russian      | ru_RU  |
| Sinhala      | si_LK  | Turkish      | tr_TR  | Vietnamese   | vi_VN  |
| Chinese (Simplified) | zh_CN | Chinese (Traditional) | zh_TW | Afrikaans | af_ZA |
| Azerbaijani  | az_AZ  | Bengali      | bn_IN  | Persian      | fa_IR  |
| Hebrew       | he_IL  | Croatian     | hr_HR  | Indonesian   | id_ID  |
| Georgian     | ka_GE  | Khmer        | km_KH  | Macedonian   | mk_MK  |
| Malayalam    | ml_IN  | Mongolian    | mn_MN  | Marathi      | mr_IN  |
| Polish       | pl_PL  | Pashto       | ps_AF  | Portuguese   | pt_XX  |
| Swedish      | sv_SE  | Swahili      | sw_KE  | Tamil        | ta_IN  |
| Telugu       | te_IN  | Thai         | th_TH  | Tagalog      | tl_XX  |
| Ukrainian    | uk_UA  | Urdu         | ur_PK  | Xhosa        | xh_ZA  |
| Galician | gl_ES | Slovene | sl_SI |  |  |

---

## 📦 Other Options

| Option | Description |
|--------|-------------|
| `--asr_backend` | Default: `funasr`. ASR backend to use. Options: `funasr` or `whisper`. |
| `--funasr_model` | Default: `auto`. FunASR model to use (only with `--asr_backend funasr`). Options: `auto` (detect automatically), `zh` (Chinese), `en` (English). |
| `--whisper_model` | Default: `base`. Whisper model size to use (only with `--asr_backend whisper`). Options: `tiny`, `base`, `small`, `medium`, `large`. |
| `--translator_backend` | Default: `llama`. Translation backend to use. Options: `llama` (local Llama2 model) or `googletrans` (Google Translate API). |
| `--output_dir, -o` | Default: `subtitled/`. Directory where the resulting subtitled videos and `.srt` files will be saved. |
| `--srt_only` | Default: `false`. If set to `true`, only the `.srt` subtitle file will be generated without creating a subtitled video. Useful for manual subtitle editing or external video processing pipelines. |

### Example

```bash
# Use FunASR (default)
auto_subtitle_cli /path/to/video.mp4

# Explicitly use FunASR with Chinese model
auto_subtitle_cli /path/to/video.mp4 --asr_backend funasr --funasr_model zh

# Use Whisper with base model
auto_subtitle_cli /path/to/video.mp4 --asr_backend whisper --whisper_model base

# Use Whisper with medium model and translate to Chinese (Simplified)
auto_subtitle_cli /path/to/video.mp4 --asr_backend whisper --whisper_model medium --translate_to zh_CN

# Use Google Translate for translation
auto_subtitle_cli /path/to/video.mp4 --translator_backend googletrans

# Save output to a custom directory
auto_subtitle_cli /path/to/video.mp4 --output_dir results/

# Generate only .srt file (no video overlay)
auto_subtitle_cli /path/to/video.mp4 --srt_only true
```

---

## 🚀 Development Workflow

The development process of this project adopted a collaborative model between
Gemini CLI and Claude Code. Gemini is responsible for task coordination and
planning, while delegating specific development tasks such as code writing and
modification to Claude Code.

You can refer to the
[Gemini-Claude Integration Template Project](link/to/your/gemini_claude_template_repo)
to understand this efficient development model.

---

## 📘 Command-line Help

To view all available options:

```bash
auto_subtitle_cli --help
```

---

## ⚖️ License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for more details.
