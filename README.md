# Automatically subtitle and translate your video

This repository uses `ffmpeg` and [OpenAI's Whisper](https://openai.com/blog/whisper) to automatically generate and overlay subtitles on any video. Then, it uses [pretrained Llama2](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate) to translate the subtitles to 50 languages.

##### [Demo Video](https://youtu.be/vkvTpmQ7M48?si=qQLvYzwtsQ4djo4K)
<p align=“center” width=“100%”>
    <img src="https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/85a41810-75ac-44f8-9b75-35c599032619" width="49%">
    <img src="https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/88d42ad7-da9f-4749-9923-4ec9fc9ed040" width="49%">
    <img src="https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/1c255fae-a1c5-4cb1-a60c-87a6aabfcf04" width="49%">
    <img src="https://github.com/YJ-20/auto-subtitle-llama/assets/68987494/91ad2860-18a7-460c-91e6-011265308433" width="49%">
</p>

## Installation

To get started, you'll need Python 3.7 or newer. Install the binary by running the following command:

    pip install git+https://github.com/YJ-20/auto-subtitle-llama

You'll also need to install [`ffmpeg`](https://ffmpeg.org/), which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
```

## Usage

The following command will generate a `subtitled/video.mp4` file contained the input video with overlayed subtitles.

    auto_subtitle_llama /path/to/video.mp4 -o subtitled/

The default setting (which selects the `turbo` model) works well for transcribing English. You can optionally use a bigger model for better results (especially with other languages). The available models are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `turbo`.

    auto_subtitle_llama /path/to/video.mp4 --model medium

Adding `--translate_to language_code` will translate the subtitles into one of the 50 languages:

    auto_subtitle_llama /path/to/video.mp4 --translate_to language_code


Language Code
Language |Arabic|Czech|German|English|Spanish|Estonian|Finnish|French|Gujarati|Hindi|Italian|Japanese|Kazakh|Korean|Lithuanian|Latvian|Burmese|Nepali|Dutch|Romanian|Russian|Sinhala|Turkish|Vietnamese|Chinese|Afrikaans|Azerbaijani|Bengali|Persian|Hebrew|Croatian|Indonesian|Georgian|Khmer|Macedonian|Malayalam|Mongolian|Marathi|Polish|Pashto|Portuguese|Swedish|Swahili|Tamil|Telugu|Thai|Tagalog|Ukrainian|Urdu|Xhosa|Galician|Slovene
:--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
Code |ar_AR|cs_CZ|de_DE|en_XX|es_XX|et_EE|fi_FI|fr_XX|gu_IN|hi_IN|it_IT|ja_XX|kk_KZ|ko_KR|lt_LT|lv_LV|my_MM|ne_NP|nl_XX|ro_RO|ru_RU|si_LK|tr_TR|vi_VN|zh_CN|af_ZA|az_AZ|bn_IN|fa_IR|he_IL|hr_HR|id_ID|ka_GE|km_KH|mk_MK|ml_IN|mn_MN|mr_IN|pl_PL|ps_AF|pt_XX|sv_SE|sw_KE|ta_IN|te_IN|th_TH|tl_XX|uk_UA|ur_PK|xh_ZA|gl_ES|sl_SI

Run the following to view all available options:

    auto_subtitle_llama --help

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.


 