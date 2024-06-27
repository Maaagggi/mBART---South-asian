from chatbot import generate_response
import warnings
import time
import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
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


def detect_and_set_language(user_input):
    """
    Detects the language of the user input, sets the source language code,
    and retrieves the language name. Handles potential exceptions.

    Args:
        user_input: The user's input text.

    Returns:
        A tuple containing:
            - detected_lang: The detected language code (or None if unsupported).
            - source_language_code: The source language code for translation.
            - detected_language_name: The full language name (or None).
    """

    try:
        detected_lang = detect(user_input)
        if detected_lang in supported_languages:
            source_language_code = supported_languages[detected_lang]
            detected_language_name = language_names.get(detected_lang)
            return detected_lang, source_language_code, detected_language_name
        else:
            # Handle unsupported language case (optional)
            print(f"Language detection failed: Unsupported language - {detected_lang}")
            return None, None, None
    except LangDetectException:
        print("Language detection failed.")
        return None, None, None


def translate_and_respond(translated_text, source_language_code):
    """
    Translates the input text to the original language, generates a chatbot response,
    and translates the response back. Handles potential translation and response generation errors.

    Args:
        translated_text: The translated text in English.

    Returns:
        A dictionary containing:
            - 'translated_reply' (optional): The translated chatbot response.
            - 'error' (optional): An error message if translation or response generation fails.
    """

    try:
        # Chatbot response using Blenderbot
        chatbot_response = generate_response(translated_text)

        # Translate response back to the original language
        translated_reply, _ = translate_mbart(chatbot_response, 'en_XX', source_language_code,
                                              max_length=512, batch_size=8)
        return {'translated_reply': translated_reply}
    except Exception as e:  # Catch any potential errors during translation or response generation
        return {'error': f"Error during translation or response generation: {str(e)}"}


if __name__ == "__main__":
    setup_nltk()

    while True:
        start_time = time.time()  # Start overall execution time measurement
        print("-------------------")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending Conversation...")
            break

        # Detect language and set variables
        detected_lang, source_language_code, detected_language_name = detect_and_set_language(user_input)

        if detected_lang:
            print(f"Detected language: {detected_language_name}")

            # Translate to English and perform further processing
            translated_text, translation_time = translate_mbart(user_input, source_language_code,
                                                                target_lang='en_XX', max_length=512, batch_size=8)

            if translated_text:
                # Call translate_and_respond with translated text
                response = translate_and_respond(translated_text, source_language_code)
