from backend.chatbot import generate_response
import warnings
import time
import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from backend.translation import translate_mbart, supported_languages

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

    while True:
        start_time = time.time()  # Start overall execution time measurement

        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending Conversation...")
            break

        # Detect language
        language_detection_start = time.time()
        detected_lang = detect_language(user_input)
        language_detection_end = time.time()

        if detected_lang and detected_lang in supported_languages:
            source_language_code = supported_languages[detected_lang]
            detected_language_name = language_names.get(detected_lang, detected_lang)
            print(f"Detected language: {detected_language_name}")

            # Translate to English
            translation_start = time.time()
            translated_text, translation_time = translate_mbart(user_input, source_language_code,
                                                                target_lang='en_XX', max_length=512, batch_size=8)
            translation_end = time.time()

            if translated_text:
                print(f"Translated text: {translated_text}")

                # Chatbot response using Blenderbot
                chatbot_start = time.time()
                chatbot_response = generate_response(translated_text)
                chatbot_end = time.time()

                # Print chatbot response and translated reply together
                print(f"Chatbot's reply in English: {chatbot_response}")
                print(f"Translated chatbot's reply in {detected_language_name}: ", end='')  # No newline after

                # Translate chatbot's reply back (optional)
                translated_reply, _ = translate_mbart(chatbot_response, 'en_XX', source_language_code,
                                                      max_length=512, batch_size=8)
                print(translated_reply)  # Print on the same line

                # Print individual task execution times
                print(f"Language detection time: {(language_detection_end - language_detection_start):.2f} seconds")
                print(f"Translation time: {translation_time:.2f} seconds")
                print(f"Chatbot response time: {(chatbot_end - chatbot_start):.2f} seconds")

            else:
                print("Translation failed.")
        else:
            print("Language detection failed - Unsupported language.")

        # Print total execution time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
