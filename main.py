import warnings
import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from chatbot import chatbot_reply
from translation import translate_mbart, supported_languages

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Ensure the language detection produces consistent results
DetectorFactory.seed = 0

# Mapping of language codes to full language names
language_names = {
    'gu': 'Gujarati', 'hi': 'Hindi', 'ne': 'Nepali', 'bn': 'Bengali', 'si': 'Sinhala',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu'
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


if __name__ == "__main__":
    setup_nltk()

    text_to_translate = "ஹாய், இன்று எப்படி இருக்கிறீர்கள்? நான் இன்று காலை உணவு சாப்பிடவில்லை."

    # Detect the language of the input text
    detected_lang = detect_language(text_to_translate)

    if detected_lang and detected_lang in supported_languages:
        source_language_code = supported_languages[detected_lang]
        detected_language_name = language_names.get(detected_lang, detected_lang)
        print(f"Detected language: {detected_language_name}")

        # Translate the detected language text to English
        translated_text, translation_time = translate_mbart(text_to_translate, source_language_code,
                                                            target_lang='en_XX',  # Translate to English
                                                            max_length=512, batch_size=8)

        if translated_text:
            print(f"Translated text: {translated_text}")
            print(f"Translation time: {translation_time:.2f} seconds")

            # Use translated_text as input to the chatbot function
            chatbot_response = chatbot_reply(translated_text)
            print(f"Chatbot's reply in English: {chatbot_response}")

            # Translate the chatbot's reply back to the original language
            translated_reply, _ = translate_mbart(chatbot_response, 'en_XX', source_language_code,
                                                  max_length=512, batch_size=8)
            print(f"Translated chatbot's reply in {detected_language_name}: {translated_reply}")

        else:
            print("Translation failed.")
    else:
        print("Language detection failed - Unsupported language.")
