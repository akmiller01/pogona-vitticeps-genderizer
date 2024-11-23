import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from collections import Counter


def main():
    dataset = load_dataset('imagefolder', data_dir='raw_images', split='train')
    dataset = dataset.add_column('class_label', dataset['label'])

    count = Counter()
    count.update(dataset['label'])
    print(count)

    dataset = dataset.class_encode_column('class_label').train_test_split(
        test_size=0.2,
        stratify_by_column="class_label",
        shuffle=True,
        seed=1337
    )
    dataset = dataset.remove_columns(['class_label'])
    dataset.push_to_hub('alex-miller/pogona-vitticeps-gender')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()