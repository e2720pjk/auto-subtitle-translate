import os
import ffmpeg
import argparse
import warnings
import tempfile
import librosa
import numpy as np
import asyncio
import re
import hashlib
from collections import OrderedDict
from funasr import AutoModel
import whisper
from .utils import *
from typing import List, Tuple
from tqdm import tqdm

# Uncomment below and comment "from .utils import *", if executing cli.py directly
import sys
sys.path.append(".")
from auto_subtitle_cli.utils import *

# deal with huggingface tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Language mapping from translate_to codes to googletrans language codes
TRANSLATE_TO_GOOGLETRANS_MAPPING = {
    'ar_AR': 'ar',  # Arabic
    'cs_CZ': 'cs',  # Czech
    'de_DE': 'de',  # German
    'en_XX': 'en',  # English
    'es_XX': 'es',  # Spanish
    'et_EE': 'et',  # Estonian
    'fi_FI': 'fi',  # Finnish
    'fr_XX': 'fr',  # French
    'gu_IN': 'gu',  # Gujarati
    'hi_IN': 'hi',  # Hindi
    'it_IT': 'it',  # Italian
    'ja_XX': 'ja',  # Japanese
    'kk_KZ': 'kk',  # Kazakh
    'ko_KR': 'ko',  # Korean
    'lt_LT': 'lt',  # Lithuanian
    'lv_LV': 'lv',  # Latvian
    'my_MM': 'my',  # Burmese
    'ne_NP': 'ne',  # Nepali
    'nl_XX': 'nl',  # Dutch
    'ro_RO': 'ro',  # Romanian
    'ru_RU': 'ru',  # Russian
    'si_LK': 'si',  # Sinhala
    'tr_TR': 'tr',  # Turkish
    'vi_VN': 'vi',  # Vietnamese
    'zh_CN': 'zh-cn',  # Chinese Simplified
    'zh_TW': 'zh-tw',  # Chinese Traditional
    'af_ZA': 'af',  # Afrikaans
    'az_AZ': 'az',  # Azerbaijani
    'bn_IN': 'bn',  # Bengali
    'fa_IR': 'fa',  # Persian
    'he_IL': 'he',  # Hebrew
    'hr_HR': 'hr',  # Croatian
    'id_ID': 'id',  # Indonesian
    'ka_GE': 'ka',  # Georgian
    'km_KH': 'km',  # Khmer
    'mk_MK': 'mk',  # Macedonian
    'ml_IN': 'ml',  # Malayalam
    'mn_MN': 'mn',  # Mongolian
    'mr_IN': 'mr',  # Marathi
    'pl_PL': 'pl',  # Polish
    'ps_AF': 'ps',  # Pashto
    'pt_XX': 'pt',  # Portuguese
    'sv_SE': 'sv',  # Swedish
    'sw_KE': 'sw',  # Swahili
    'ta_IN': 'ta',  # Tamil
    'te_IN': 'te',  # Telugu
    'th_TH': 'th',  # Thai
    'tl_XX': 'tl',  # Tagalog
    'uk_UA': 'uk',  # Ukrainian
    'ur_PK': 'ur',  # Urdu
    'xh_ZA': 'xh',  # Xhosa
    'gl_ES': 'gl',  # Galician
    'sl_SI': 'sl',  # Slovene
}

# Global cache for translations with LRU-like behavior
TRANSLATION_CACHE = OrderedDict()
MAX_CACHE_SIZE = 1000

def get_cache_key(text: str, target_lang: str) -> str:
    """Generate a cache key from text and target language."""
    content = f"{text}|{target_lang}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_from_cache(cache_key: str) -> str:
    """Get translation from cache with LRU behavior."""
    if cache_key in TRANSLATION_CACHE:
        # Move to end (most recently used)
        value = TRANSLATION_CACHE.pop(cache_key)
        TRANSLATION_CACHE[cache_key] = value
        return value
    return None

def put_in_cache(cache_key: str, translation: str):
    """Put translation in cache with size limit."""
    if cache_key in TRANSLATION_CACHE:
        # Update existing entry
        TRANSLATION_CACHE.pop(cache_key)
    elif len(TRANSLATION_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        TRANSLATION_CACHE.popitem(last=False)

    TRANSLATION_CACHE[cache_key] = translation

def split_long_text(text: str, max_length: int = 15000) -> List[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence is too long, split by clauses
        if len(sentence) > max_length:
            clauses = re.split(r'[,;:]+', sentence)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue

                if len(clause) > max_length:
                    # Split by words as last resort
                    words = clause.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) > max_length:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                # Single word is too long, force split
                                chunks.append(word[:max_length])
                                word = word[max_length:]
                                word_chunk = word
                        else:
                            word_chunk = word_chunk + " " + word if word_chunk else word
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                else:
                    if len(current_chunk + " " + clause) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = clause
                        else:
                            chunks.append(clause)
                    else:
                        current_chunk = current_chunk + " " + clause if current_chunk else clause
        else:
            if len(current_chunk + " " + sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    chunks.append(sentence)
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def merge_short_texts(text_batch: List[str], min_length: int = 5, target_length: int = 50) -> Tuple[List[str], List[List[int]]]:
    """Merge short texts and return mapping for reconstruction."""
    if not text_batch:
        return [], []

    merged_texts = []
    mapping = []  # List of lists containing original indices for each merged text

    current_merged = ""
    current_indices = []

    for i, text in enumerate(text_batch):
        text = text.strip()

        # If text is very short, try to merge
        if len(text) < min_length and len(current_merged + " " + text) <= target_length:
            if current_merged:
                current_merged += " " + text
            else:
                current_merged = text
            current_indices.append(i)
        else:
            # If we have accumulated short texts, add them as a merged group
            if current_merged:
                merged_texts.append(current_merged)
                mapping.append(current_indices)
                current_merged = ""
                current_indices = []

            # Add current text as individual item
            merged_texts.append(text)
            mapping.append([i])

    # Don't forget the last accumulated group
    if current_merged:
        merged_texts.append(current_merged)
        mapping.append(current_indices)

    return merged_texts, mapping

def reconstruct_from_merged(merged_translations: List[str], mapping: List[List[int]], original_count: int) -> List[str]:
    """Reconstruct original structure from merged translations."""
    result = [""] * original_count

    for merged_text, indices in zip(merged_translations, mapping):
        if len(indices) == 1:
            # Single text, direct assignment
            result[indices[0]] = merged_text
        else:
            # Multiple texts were merged, need to split
            # Simple approach: split by common separators and distribute
            parts = re.split(r'[.!?]\s+', merged_text)

            # If we have exactly the right number of parts
            if len(parts) == len(indices):
                for idx, part in zip(indices, parts):
                    result[idx] = part.strip()
            else:
                # Fallback: distribute evenly or use the whole text for each
                for idx in indices:
                    result[idx] = merged_text

    return result

def detect_language_funasr(audio_path):
    """Simple language detection based on common characters in transcription."""
    # Try Chinese model first for a small sample
    try:
        audio_data, _ = librosa.load(audio_path, sr=16000, duration=30)  # Only use first 30 seconds

        zh_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common"
        )

        result = zh_model.generate(
            audio_data,
            return_spk_res=False,
            sentence_timestamp=True,
            return_raw_text=True,
            is_final=True,
            hotword="",
            pred_timestamp=False,
            en_post_proc=False,
            cache={}
        )

        text = result[0]['text']
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.isalnum()])

        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        else:
            return "en"
    except:
        return "en"  # Default to English if detection fails

def convert_funasr_to_segments(sentence_info):
    """Convert FunASR sentence_info to Whisper-like segments format."""
    segments = []
    for i, sent in enumerate(sentence_info):
        if sent['timestamp']:
            start_ms = sent['timestamp'][0][0]
            end_ms = sent['timestamp'][-1][1]
            segments.append({
                'id': i,
                'start': start_ms / 1000.0,  # Convert ms to seconds
                'end': end_ms / 1000.0,      # Convert ms to seconds
                'text': sent['text']
            })
    return segments

async def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--asr_backend", type=str, default="funasr", choices=["funasr", "whisper"],
                        help="ASR backend to use: 'funasr' or 'whisper'")
    parser.add_argument("--funasr_model", default="auto",
                        choices=["auto", "zh", "en"], help="FunASR model to use (auto: detect automatically, zh: Chinese, en: English)")
    parser.add_argument("--whisper_model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="subtitled", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=True,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")
    parser.add_argument("--initial_prompt", type=str, default=None, help="Optional text to provide as a prompt for the first window to guide the model's style or continue a previous transcription.")

    parser.add_argument("--translate_to", type=str, default=None, choices=['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN', 'zh_TW', 'af_ZA', 'az_AZ', 'bn_IN', 'fa_IR', 'he_IL', 'hr_HR', 'id_ID', 'ka_GE', 'km_KH', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN', 'pl_PL', 'ps_AF', 'pt_XX', 'sv_SE', 'sw_KE', 'ta_IN', 'te_IN', 'th_TH', 'tl_XX', 'uk_UA', 'ur_PK', 'xh_ZA', 'gl_ES', 'sl_SI'],
    help="Final target language code; Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese Simplified (zh_CN), Chinese Traditional (zh_TW), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)")
    parser.add_argument("--translator_backend", type=str, default="llama", choices=["llama", "googletrans"],
                        help="Translation backend to use: 'llama' for local Llama2 model or 'googletrans' for Google Translate API")
    
    args = parser.parse_args().__dict__
    asr_backend: str = args.pop("asr_backend")
    funasr_model_name: str = args.pop("funasr_model")
    whisper_model_name: str = args.pop("whisper_model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    translate_to: str = args.pop("translate_to")
    translator_backend: str = args.pop("translator_backend")
    
    os.makedirs(output_dir, exist_ok=True)

    # Initialize ASR model based on backend choice
    asr_model = None
    model_language = None

    if asr_backend == "funasr":
        # Initialize FunASR model based on language
        if funasr_model_name == "auto":
            # Detect language automatically using the first audio file
            first_audio_path = list(get_audio(args["video"]).values())[0]
            detected_lang = detect_language_funasr(first_audio_path)
            funasr_model_name = detected_lang

        if funasr_model_name == "zh" or (language != "auto" and language in ["zh", "zh-cn", "zh-tw", "chinese"]):
            asr_model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common"
            )
            model_language = "zh"
        else:
            asr_model = AutoModel(
                model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common"
            )
            model_language = "en"
    elif asr_backend == "whisper":
        # Initialize Whisper model
        asr_model = whisper.load_model(whisper_model_name)
        model_language = None  # Will be detected dynamically for Whisper

    if language != "auto":
        args["language"] = language
    audios = get_audio(args.pop("video"))
    subtitles, detected_language = await get_subtitles(
        audios,
        output_srt or srt_only,
        output_dir,
        asr_model,
        asr_backend,
        model_language,
        args,
        translate_to=translate_to,
        translator_backend=translator_backend
    )

    if srt_only:
        return
    
    _translated_to = ""
    if translate_to:
        # for filename
        _translated_to = f"2{translate_to.split('_')[0]}"
        
    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_subtitled_{detected_language}{_translated_to}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="FallbackName=NanumGothic,OutlineColour=&H40000000,BorderStyle=3", charenc="UTF-8"), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


async def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, asr_model, asr_backend: str, model_language: str, args: dict, translate_to: str = None, translator_backend: str = "llama") -> Tuple[dict, str]:
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")

        if asr_backend == "funasr":
            print("[Step1] detect language (FunASR)")

            # Load audio with librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            if len(audio_data.shape) == 2:  # multi-channel
                audio_data = audio_data[:, 0]  # take first channel

            detected_language = model_language
            current_lang = LANG_CODE_MAPPER.get(detected_language, [])

            print("[Step2] transcribe (FunASR)")
            if detected_language != "en" and translate_to is not None and translate_to not in current_lang:
                args["task"] = "translate"
                print(f"transcribe-task changed for llama translator")

            # Use FunASR for transcription
            rec_result = asr_model.generate(
                audio_data,
                return_spk_res=False,
                sentence_timestamp=True,
                return_raw_text=True,
                is_final=True,
                hotword="",
                pred_timestamp=model_language=='en',
                en_post_proc=model_language=='en',
                cache={}
            )

            # Convert FunASR result to Whisper-like segments format
            segments = convert_funasr_to_segments(rec_result[0]['sentence_info'])
            result = {
                "text": rec_result[0]['text'],
                "segments": segments
            }

        elif asr_backend == "whisper":
            print("[Step1] detect language (Whisper)")
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            # make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio, asr_model.dims.n_mels).to(asr_model.device)
            # detect the spoken language
            _, probs = asr_model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            current_lang = LANG_CODE_MAPPER.get(detected_language, [])

            print("[Step2] transcribe (Whisper)")
            if detected_language != "en" and translate_to is not None and translate_to not in current_lang:
                args["task"] = "translate"
                print(f"transcribe-task changed for llama translator")
            result = asr_model.transcribe(audio_path, **args)
        
        if translate_to is not None and translate_to not in current_lang:
            if translator_backend == "googletrans":
                print("[Step3] translate (GoogleTrans)")
                text_batch = get_text_batch(segments=result["segments"])
                translated_batch = await translates_googletrans(translate_to=translate_to, text_batch=text_batch)
                result["segments"] = replace_text_batch(segments=result["segments"], translated_batch=translated_batch)
                print(f"translated to {translate_to} using GoogleTrans")
            else:  # llama backend
                if translate_to == "zh_TW":
                    print("[Warning] Llama2 does not directly support Traditional Chinese (zh_TW).")
                    print("Please use '--translator_backend googletrans' for Traditional Chinese translation.")
                    print("Keeping original transcription without translation.")
                else:
                    print("[Step3] translate (Llama2)")
                    text_batch = get_text_batch(segments=result["segments"])
                    translated_batch = translates_llama(translate_to=translate_to, text_batch=text_batch)
                    result["segments"] = replace_text_batch(segments=result["segments"], translated_batch=translated_batch)
                    print(f"translated to {translate_to} using Llama2")
        
        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
        print(f"srt file is saved: {srt_path}")
        subtitles_path[path] = srt_path

    return subtitles_path, detected_language

def translates_llama(translate_to: str, text_batch: List[str], max_batch_size: int = 32):
    model, tokenizer = load_translator()
    
    # split text_batch into max_batch_size
    divided_text_batches = [text_batch[i:i+max_batch_size] for i in range(0, len(text_batch), max_batch_size)]
    
    translated_batch = []
    
    for batch in tqdm(divided_text_batches, desc="batch translate"):
        model_inputs = tokenizer(batch, return_tensors="pt", padding=True)
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[translate_to]
        )
        translated_batch.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    
    return translated_batch

async def translates_googletrans(translate_to: str, text_batch: List[str], max_batch_size: int = 100):
    try:
        from googletrans import Translator
    except ImportError:
        raise ImportError("googletrans library is required for googletrans backend. Install it with: pip install googletrans")

    # Get the googletrans language code
    target_lang = TRANSLATE_TO_GOOGLETRANS_MAPPING.get(translate_to)
    if not target_lang:
        raise ValueError(f"Unsupported language code for googletrans: {translate_to}")

    # Step 1: Merge short texts to reduce API calls
    merged_texts, merge_mapping = merge_short_texts(text_batch, min_length=5, target_length=50)
    print(f"Merged {len(text_batch)} texts into {len(merged_texts)} chunks for translation")

    # Step 2: Process each merged text for translation
    processed_texts = []
    split_mappings = []  # Track how each merged text was split

    for merged_text in merged_texts:
        # Handle long texts by splitting them
        if len(merged_text) > 15000:
            chunks = split_long_text(merged_text, max_length=15000)
            processed_texts.extend(chunks)
            split_mappings.append(list(range(len(processed_texts) - len(chunks), len(processed_texts))))
        else:
            processed_texts.append(merged_text)
            split_mappings.append([len(processed_texts) - 1])

    # Step 3: Check cache and prepare texts for translation
    translations = [""] * len(processed_texts)
    texts_to_translate = []
    translate_indices = []

    for i, text in enumerate(processed_texts):
        cache_key = get_cache_key(text.strip(), target_lang)
        cached_translation = get_from_cache(cache_key)

        if cached_translation is not None:
            translations[i] = cached_translation
        else:
            texts_to_translate.append(text)
            translate_indices.append(i)

    print(f"Found {len(processed_texts) - len(texts_to_translate)} cached translations, need to translate {len(texts_to_translate)} texts")

    # Step 4: Translate uncached texts
    if texts_to_translate:
        async with Translator() as translator:
            # Split texts_to_translate into max_batch_size chunks for better reliability
            divided_text_batches = [texts_to_translate[i:i+max_batch_size] for i in range(0, len(texts_to_translate), max_batch_size)]
            translated_results = []

            for batch in tqdm(divided_text_batches, desc="googletrans batch translate"):
                try:
                    # googletrans supports batch translation
                    batch_translations = await translator.translate(batch, dest=target_lang)

                    # Handle both single translation and batch translations
                    if isinstance(batch_translations, list):
                        translated_results.extend([t.text for t in batch_translations])
                    else:
                        translated_results.append(batch_translations.text)

                except Exception as e:
                    print(f"Warning: GoogleTrans translation error: {e}")
                    print("Falling back to individual translation...")

                    # Fallback: translate one by one
                    for text in batch:
                        try:
                            result = await translator.translate(text, dest=target_lang)
                            translated_results.append(result.text)
                        except Exception as individual_error:
                            print(f"Error translating text: {text[:50]}... Error: {individual_error}")
                            # If individual translation fails, keep original text
                            translated_results.append(text)

            # Step 5: Store results in cache and update translations array
            for i, (text, translation) in enumerate(zip(texts_to_translate, translated_results)):
                cache_key = get_cache_key(text.strip(), target_lang)
                put_in_cache(cache_key, translation)
                translations[translate_indices[i]] = translation

    # Step 6: Reconstruct merged texts from split chunks
    merged_translations = []
    for split_indices in split_mappings:
        if len(split_indices) == 1:
            # Single chunk, use directly
            merged_translations.append(translations[split_indices[0]])
        else:
            # Multiple chunks, combine them
            combined = " ".join(translations[idx] for idx in split_indices)
            merged_translations.append(combined)

    # Step 7: Reconstruct original structure from merged translations
    final_translations = reconstruct_from_merged(merged_translations, merge_mapping, len(text_batch))

    return final_translations


def cli_main():
    """Entry point for the console script."""
    asyncio.run(main())

if __name__ == '__main__':
    cli_main()
