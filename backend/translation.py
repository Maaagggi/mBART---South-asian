import time
import warnings
import logging
import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

DetectorFactory.seed = 0

supported_languages = {
    'gu': 'gu_IN', 'hi': 'hi_IN', 'ne': 'ne_NP', 'bn': 'bn_IN', 'si': 'si_LK',
    'ml': 'ml_IN', 'mr': 'mr_IN', 'ta': 'ta_IN', 'te': 'te_IN', 'ur': 'ur_PK'
}


def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def detect_language(text):
    try:
        detected_lang = detect(text)
        return detected_lang
    except LangDetectException:
        return None


def translate_mbart(text, source_lang, target_lang='en_XX', max_length=1024, batch_size=8):
    try:
        logging.basicConfig(filename='translation.log', level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        model_name = 'facebook/mbart-large-50-many-to-many-mmt'
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)

        tokenizer.src_lang = source_lang

        text_to_encode = [text]

        translated_chunks = []
        start_time = time.time()

        for i in range(0, len(text_to_encode), batch_size):
            batch = text_to_encode[i:i + batch_size]
            encoded_texts = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True,
                                      max_length=max_length)

            generated_tokens = model.generate(**encoded_texts,
                                              forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                                              max_length=max_length)

            translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translated_chunks.extend(translated_texts)

        end_time = time.time()
        translation_time = end_time - start_time

        logger.debug(f"Translated text: {translated_chunks[0]}")

        return translated_chunks[0], translation_time
    except KeyError:
        print(f"Translation failed: Unsupported language code '{source_lang}'")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
