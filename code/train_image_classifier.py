import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator, TrainingArguments, Trainer


checkpoint = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size['shortest_edge']
    if 'shortest_edge' in image_processor.size
    else (image_processor.size['height'], image_processor.size['width'])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
def transforms(examples):
    examples['pixel_values'] = [_transforms(img.convert('RGB')) for img in examples['image']]
    del examples['image']
    return examples


accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main():
    dataset = load_dataset('alex-miller/pogona-vitticeps-gender')

    labels = dataset['train'].features['label'].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    
    dataset = dataset.with_transform(transforms)
    data_collator = DefaultDataCollator()
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir='pogona-vitticeps-gender',
        remove_unused_columns=False,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=True,
        report_to='tensorboard'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()



if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()