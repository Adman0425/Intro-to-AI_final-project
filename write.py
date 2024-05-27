import argparse
from transformers import TFGPT2LMHeadModel, TextGenerationPipeline
import tensorflow as tf
from train import load_tokenizer
from train import config

def parse_arguments():
    """
    Parse command line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for text generation using a trained GPT-2 model.")
    parser.add_argument('-d', '--dir', default=f"{config.model_pos}-50", help="Directory where the model is stored.")
    parser.add_argument('-max', '--max_len', type=int, default=128, help="Maximum length of the generated text.")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to handle the text generation workflow:
    1. Display welcome prompt
    2. Parse command line arguments
    3. Load the model and tokenizer
    4. Accept input string from the user
    5. Generate and display text
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Default values
    default_model_path = f"{config.model_pos}-50"
    default_max_len = 128

    # Welcome prompt
    print("## Write.py ##")
    print(f"\tBy default, it will use the model stored in -> {default_model_path}")
    print("\tUse the --dir option to specify a different model directory.")

    # Parse command line arguments
    print("==> Parsing Arguments:")
    args = parse_arguments()
    model_path = args.dir if args.dir != default_model_path else default_model_path
    max_len = args.max_len if args.max_len != default_max_len else default_max_len
    print("\tArguments parsed...")

    # Load model and tokenizer
    print("==> Loading Model & Tokenizer:")
    try:
        tokenizer = load_tokenizer("train")
        print("\tTokenizer loaded...")
        model = TFGPT2LMHeadModel.from_pretrained(model_path)
        print("\tModel loaded...")
    except OSError:
        print(f"The directory {model_path} may not exist.")
        print("Use `python write.py --dir <dir>` to specify the model path if needed.")
        exit(255)

    # Accept input from user
    print("## Input Start Sequence ##")
    input_text = input("Input: ")
    text_generator = TextGenerationPipeline(model, tokenizer)
    print("\tText generator initialized...")

    # Generate and display text
    print("==> Generating Text:")
    generated_text = text_generator(
        text_inputs=input_text,
        max_length=max_len,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.get_vocab().get("</s>", 0)
    )[0]['generated_text']
    
    print(f"## Result ##\n{generated_text}")
    print("\tText generated...")


if __name__ == '__main__':
    main()
