import os
import tensorflow as tf
from transformers import GPT2Config, GPT2Tokenizer, TextGenerationPipeline
from src.config import ProjectConfig
from src.model import TextModel
import matplotlib.pyplot as plt
import json

def read_jsonl(file_path):
    import json
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line.strip()) for line in file]
    return data

# Setting metadata and configuration
print("==> Setting Metadata:")
project_config = ProjectConfig(
    block_size=100,
    batch_size=12,
    buffer_size=1000,
    data_name="example",
    epoch_times=50
)
# Set seed for reproducibility
tf.random.set_seed(42)
print("\tMetadata set...")


def load_tokenizer(tokenizer_type: str) -> GPT2Tokenizer:
    """
    Load and configure the GPT-2 tokenizer.

    Args:
        tokenizer_type (str): Type of the tokenizer to load ('train' or 'test').

    Returns:
        GPT2Tokenizer: The configured tokenizer.
    """
    token_path = project_config.token_pos if tokenizer_type == "train" else project_config.test_token_pos
    tokenizer = GPT2Tokenizer.from_pretrained(token_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    return tokenizer


def create_dataset(tokenizer: GPT2Tokenizer, dataset_type: str) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for training or testing.

    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding the data.
        dataset_type (str): Type of the dataset to create ('train' or 'test').

    Returns:
        tf.data.Dataset: The created dataset.
    """
    # data_path = project_config.train_pos if dataset_type == "train" else project_config.test_pos
    # with open(data_path, "r", encoding='utf-8') as file:
    #     text_data = file.read().replace("\n", " ")

    data_path = project_config.train_pos if dataset_type == "train" else project_config.test_pos
    jsonl_data = read_jsonl(data_path)
    
    # 提取文本内容
    text_data = " ".join([msg['content'] for item in jsonl_data for msg in item['messages']])

    tokenized_data = tokenizer.encode(text_data)
    print("\tText Encoded...")

    # Create input and label pairs for the dataset
    examples = [tokenized_data[i:i + project_config.block_size] for i in range(0, len(tokenized_data) - project_config.block_size + 1, project_config.block_size)]
    inputs = [ex[:-1] for ex in examples]
    labels = [ex[1:] for ex in examples]

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(buffer_size=project_config.buffer_size).batch(project_config.batch_size, drop_remainder=True)
    return dataset


def visualize_comparison(train_model: TextModel, test_model: TextModel):
    """
    Visualize and compare the training and testing loss.

    Args:
        train_model (TextModel): The model trained on the training dataset.
        test_model (TextModel): The model trained on the testing dataset.
    """
    # Visualize per epoch loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_model.history.history['loss'], color='red', label='train')
    plt.plot(test_model.history.history['loss'], color='blue', label='test')
    plt.legend(loc='upper left')
    plt.title('Comparison of Test & Train Loss (Per Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{project_config.pltfigure_pos}/{project_config.data_name}-{project_config.epoch_times}-train-test-epoch.png")
    plt.show()
    plt.close()

    # Visualize per batch loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_model.batch_end_loss[:len(test_model.batch_end_loss)], color='red', label='train')
    plt.plot(test_model.batch_end_loss, color='blue', label='test')
    plt.legend(loc='upper left')
    plt.title('Comparison of Test & Train Loss (Per Batch)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(f"{project_config.pltfigure_pos}/{project_config.data_name}-{project_config.epoch_times}-train-test-batch.png")
    plt.show()
    plt.close()


def main():
    # Load tokenizers
    print("==> Loading tokenizer:")
    train_tokenizer = load_tokenizer(tokenizer_type='train')
    test_tokenizer = load_tokenizer(tokenizer_type='test')
    print("\tTokenizer loaded...")

    # Create datasets
    print("==> Making dataset:")
    train_dataset = create_dataset(tokenizer=train_tokenizer, dataset_type="train")
    test_dataset = create_dataset(tokenizer=test_tokenizer, dataset_type="test")
    print("\tDataset made...")

    # Initialize models
    print("==> Initializing model:")
    train_model = TextModel(config=project_config, tokenizer=train_tokenizer, model_name="train")
    test_model = TextModel(config=project_config, tokenizer=test_tokenizer, model_name="test")
    print("\tModel initialized...")

    # Train models
    print("==> Training model:")
    train_model.train(dataset=train_dataset)
    test_model.train(dataset=test_dataset)
    print("\tModel trained...")

    # Save models
    print("==> Saving models:")
    train_model.save("save_train_model")
    test_model.save("save_test_model")
    print("\tModels saved...")

    # Visualize results
    print("==> Visualizing results:")
    train_model.visualize()
    test_model.visualize()
    print("\tIndividual model visualized...")
    visualize_comparison(train_model=train_model, test_model=test_model)
    print("\tComparison visualized...")

    # Output training results
    print("==> Outputting training results:")
    train_model.training_output()
    test_model.training_output()
    print("\tTraining results outputted...")

    # Text generation loop
    print("==> Ready for text generation. Press Ctrl+C to exit.")
    while True:
        try:
            user_input = input("Input: ")
            text_generator = TextGenerationPipeline(model=train_model.model, tokenizer=train_tokenizer)
            generated_text = text_generator(
                text_inputs=user_input,
                max_length=128,
                do_sample=True,
                top_k=10,
                eos_token_id=train_tokenizer.get_vocab().get("</s>", 0)
            )[0]['generated_text']
            print(f"Result: {generated_text}")
        except KeyboardInterrupt:
            print("Exiting text generation...")
            break


if __name__ == "__main__":
    main()
