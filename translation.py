import time
import warnings
import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Ensure the language detection produces consistent results
DetectorFactory.seed = 0

# Supported South Asian language codes for mBART
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


def split_text(text, max_length=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def translate_mbart(text, source_lang, target_lang='en_XX', max_length=512, batch_size=8):
    try:
        # Load the tokenizer and model
        model_name = 'facebook/mbart-large-50-many-to-many-mmt'
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)

        # Set the source language for the tokenizer
        tokenizer.src_lang = source_lang

        # Split text into smaller chunks
        text_chunks = split_text(text, max_length)

        translated_chunks = []
        start_time = time.time()

        # Batch processing
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            encoded_texts = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True,
                                      max_length=max_length)

            # Perform the translation for the batch
            generated_tokens = model.generate(**encoded_texts,
                                              forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                                              max_length=max_length)

            # Decode the generated tokens
            translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translated_chunks.extend(translated_texts)

        end_time = time.time()
        translation_time = end_time - start_time

        return " ".join(translated_chunks), translation_time
    except KeyError:
        print(f"Translation failed: Unsupported language code '{source_lang}'")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


# Example usage
if __name__ == "__main__":
    setup_nltk()

    text_to_translate = ('हो। गए, उनका एक समय में बड़ा नाम था। पूरे देश में तालाब बनते थे बनाने वाले भी पूरे देश में '
                         'थे। कहीं यह विद्या जाति के विद्यालय | सिखाई जाती थी तो कहीं यह जात से हट कर एक विशेष पांत '
                         'भी जाती थी। बनाने वाले लोग कहीं एक जगह बसे मिलते थे तो कहीं -घूम कर इस काम को करते थे। I 국 '
                         'घम गजधर एक सुन्दर शब्द है, तालाब बनाने वालों को आदर के साथ याद करने के लिए। राजस्थान के कुछ '
                         'भागों में यह शब्द आज भी बाकी है। गजधर यानी जो गज को धारण करता है। और गज वही जो नापने के काम '
                         'आता है। लेकिन फिर भी समाज ने इन्हें तीन हाथ की लोहे की छड़ लेकर घूमने वाला मिस्त्री नहीं '
                         'माना। गजधर जो समाज को गहराई को नाप ले - उसे ऐसा दर्जा दिया गया है। गजधर वास्तुकार थे। '
                         'गांव-समाज हो या नगर-समाज - उसके नव निर्माण की, रख-रखाव की ज़िम्मेदारी गजधर निभाते थे। नगर '
                         'नियोजन से लेकर छोटे से छोटे निर्माण के काम गजधर के कधों पर टिके थे। वे योजना बनाते थे, '
                         'कुल काम की लागत निकालते थे, काम में लगने वाली सारी सामग्री जुटाते थे और इस सबके बदले वे '
                         'अपने जजमान से ऐसा कुछ नहीं मांग बैठते थे, जो वे दे न पाएं। लोग भी ऐसे थे कि उनसे जो कुछ '
                         'बनता, वे गजधर को भेंट कर देते। काम पूरा होने पर पारिश्रमिक के अलावा गजधर को सम्मान भी मिलता '
                         'था। सरोपा भेंट करना अब शायद सिर्फ सिख परंपरा में ही बचा समाज की गहराई नापते रहे हैं '
                         'गुणाधर')  # Example text in Hindi

    # Detect the language of the input text
    detected_lang = detect_language(text_to_translate)

    if detected_lang and detected_lang in supported_languages:
        source_language_code = supported_languages[detected_lang]
        print(f"Detected language: {detected_lang} ({source_language_code})")

        # Translate the detected language text to English
        translated_text, translation_time = translate_mbart(text_to_translate, source_language_code,
                                                            target_lang='en_XX', max_length=512, batch_size=8)

        if translated_text:
            print(f"Translated text: {translated_text}")
            print(f"Translation time: {translation_time:.2f} seconds")

        else:
            print("Translation failed.")
    else:
        print("Language detection failed or unsupported language.")
