from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings("ignore")


class SimpleTranslator:

    def __init__(self):
        """Initialize the translator with empty model and tokenizer"""
        self.model = None
        self.tokenizer = None
        self.current_language = None

    def load_language_model(self, target_language):
        # Create model name for English to target language
        model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"

        try:
            print(f"Loading model for English to {target_language.upper()}...")

            # Load the tokenizer (converts text to numbers)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

            # Load the pre-trained model
            self.model = MarianMTModel.from_pretrained(model_name)

            self.current_language = target_language
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check if the language code is valid.")

    def translate(self, text):
        if self.model is None or self.tokenizer is None:
            return "Please load a language model first!"

        try:
            # Step 1: Convert text to tokens (numbers the model understands)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)

            # Step 2: Generate translation using the model
            translated_tokens = self.model.generate(**inputs)

            # Step 3: Convert tokens back to readable text
            translated_text = self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )

            return translated_text

        except Exception as e:
            return f"Translation error: {e}"


def main():
    print("=== Simple English Translator ===\n")

    # Create translator instance
    translator = SimpleTranslator()

    # Available languages (you can add more)
    languages = {
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese'
    }

    print("Available languages:")
    for code, name in languages.items():
        print(f" {code} - {name}")

    # Get user's language choice
    while True:
        lang_code = input("\nEnter language code (or 'quit' to exit): ").lower()

        if lang_code == 'quit':
            break

        if lang_code in languages:
            # Load the model for chosen language
            translator.load_language_model(lang_code)

            # Translation loop
            while True:
                text = input(
                    f"\nEnter English text to translate to {languages[lang_code]} "
                    "(or 'back' to choose another language): "
                )

                if text.lower() == 'back':
                    break

                if text.strip():
                    # Translate the text
                    result = translator.translate(text)
                    print(f"Translation: {result}")
                else:
                    print("Please enter some text to translate.")
        else:
            print("Invalid language code. Please try again.")


# Example usage and testing
if __name__ == "__main__":
    print("Example Usage:")
    print("-" * 40)

    # Create a translator
    translator = SimpleTranslator()

    # Load French model
    translator.load_language_model('fr')

    # Test translations
    test_sentences = [
        "Hello, how are you?",
        "I love learning new languages.",
        "The weather is beautiful today.",
        "Thank you for your help."
    ]

    print("\nTest Translations to French:")
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"EN: {sentence}")
        print(f"FR: {translation}\n")

    print("Now running interactive translator...")
    print("Note: First time loading a model may take a few minutes to download.\n")

    main()
