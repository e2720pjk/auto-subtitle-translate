# 🎬 自動字幕生成與翻譯工具 FunASR / Whisper + LLaMA / Google Translate

🌐 **語言**: [English](README.md) | [繁體中文](README_zh-TW.md)

使用 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 或
[OpenAI Whisper](https://openai.com/blog/whisper) 自動為任何影片生成字幕，
透過 `ffmpeg` 嵌入字幕，並可選用基於
[LLaMA2 的多語言翻譯模型](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate)
或 [Googletrans](https://github.com/ssut/py-googletrans) 將字幕翻譯成 50 多種語言。

## 📺 [Demo Video](https://youtu.be/vkvTpmQ7M48?si=qQLvYzwtsQ4djo4K)

![自動字幕生成演示截圖 1](https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/85a41810-75ac-44f8-9b75-35c599032619)

![自動字幕生成演示截圖 2](https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/88d42ad7-da9f-4749-9923-4ec9fc9ed040)

![自動字幕生成演示截圖 3](https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/1c255fae-a1c5-4cb1-a60c-87a6aabfcf04)

![自動字幕生成演示截圖 4](https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/91ad2860-18a7-460c-91e6-011265308433)

---

## 🛠️ 安裝

請確保您使用的是 Python 3.7 或更新版本。

直接從 GitHub 安裝套件：

```bash
pip install git+https://github.com/e2720pjk/auto-subtitle-translate
```

**FunASR 和 Google Translate 的額外相依套件：**

```bash
pip install funasr librosa torch torchaudio numpy scipy soundfile googletrans
```

### 安裝 `ffmpeg`

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (使用 Homebrew)
brew install ffmpeg

# Windows (使用 Chocolatey)
choco install ffmpeg
```

---

## 🚀 使用方法

### 僅轉錄字幕（不翻譯）

```bash
auto_subtitle_cli /path/to/video.mp4
```

### 轉錄並翻譯字幕

將字幕翻譯成其他語言：

```bash
# 使用 LLaMA2 後端（預設）
auto_subtitle_cli /path/to/video.mp4 --translate_to ko_KR

# 使用 Google Translate 後端（建議用於繁體中文且準確度更高）
auto_subtitle_cli /path/to/video.mp4 --translate_to zh_TW \
  --translator_backend googletrans
```

---

## 🌐 支援的翻譯語言

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

## 📦 其他選項

| 選項 | 說明 |
|------|------|
| `--asr_backend` | 預設值：`funasr`。要使用的 ASR 後端。選項：`funasr` 或 `whisper`。 |
| `--funasr_model` | 預設值：`auto`。要使用的 FunASR 模型（僅當 `--asr_backend` 為 `funasr` 時）。選項：`auto`（自動偵測）、`zh`（中文）、`en`（英文）。 |
| `--whisper_model` | 預設值：`base`。要使用的 Whisper 模型大小（僅當 `--asr_backend` 為 `whisper` 時）。選項：`tiny`、`base`、`small`、`medium`、`large`。 |
| `--translator_backend` | 預設值：`llama`。要使用的翻譯後端。選項：`llama`（本地 Llama2 模型）或 `googletrans`（Google Translate API）。|
| `--output_dir, -o` | 預設值：`subtitled/`。儲存產生的字幕影片和 `.srt` 檔案的目錄。 |
| `--srt_only` | 預設值：`false`。若設為 `true`，只會產生 `.srt` 字幕檔案而不建立嵌入字幕的影片。適用於手動編輯字幕或外部影片處理流程。|

### 範例

```bash
# 使用 FunASR (預設)
auto_subtitle_cli /path/to/video.mp4

# 明確指定 FunASR 並使用中文模型
auto_subtitle_cli /path/to/video.mp4 --asr_backend funasr --funasr_model zh

# 使用 Whisper 並使用 base 模型
auto_subtitle_cli /path/to/video.mp4 --asr_backend whisper --whisper_model base

# 使用 Whisper 並使用 medium 模型，同時翻譯成簡體中文
auto_subtitle_cli /path/to/video.mp4 --asr_backend whisper --whisper_model medium --translate_to zh_CN

# 使用 Google Translate 進行翻譯
auto_subtitle_cli /path/to/video.mp4 --translator_backend googletrans

# 將輸出儲存到自訂目錄
auto_subtitle_cli /path/to/video.mp4 --output_dir results/

# 只產生 .srt 檔案（不嵌入影片）
auto_subtitle_cli /path/to/video.mp4 --srt_only true
```

---

## 🚀 開發流程

此專案的開發過程採用了 Gemini CLI 與 Claude Code 的協作模式。
Gemini 負責任務的協調與規劃，並將程式碼編寫、修改等具體開發任務委派給 Claude Code 執行。

您可以參考
[Gemini-Claude 整合模板專案](link/to/your/gemini_claude_template_repo)
來了解這種高效的開發模式。

---

## 📘 命令列說明

查看所有可用選項：

```bash
auto_subtitle_cli --help
```

---

## ⚖️ 授權條款

本專案採用 MIT 授權條款。
詳情請參閱 [LICENSE](LICENSE) 檔案。
