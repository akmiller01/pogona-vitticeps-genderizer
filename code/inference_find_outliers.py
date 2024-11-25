from transformers import AutoImageProcessor
import torch
from transformers import AutoModelForImageClassification
from datasets import load_dataset
from scipy.special import softmax
from tqdm import tqdm

global image_processor
global model
image_processor = AutoImageProcessor.from_pretrained('alex-miller/pogona-vitticeps-gender')
model = AutoModelForImageClassification.from_pretrained("alex-miller/pogona-vitticeps-gender")


def inference(image):
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    logits = softmax(logits)

    predicted_label = logits.argmax(-1).item()
    predicted_confidence = logits.max()
    return predicted_label, predicted_confidence


def main():
    dataset = load_dataset('imagefolder', data_dir='raw_images', split='train')

    for example in tqdm(dataset):
        image = example['image']
        label = example['label']
        predicted_label, predicted_confidence = inference(image)
        if predicted_label != label and predicted_confidence >= 0.5:
            print(f"""
            Filename: {image.filename}
            Correct label: {model.config.id2label[label]}
            Predicted label: {model.config.id2label[predicted_label]}
            Predicted confidence: {predicted_confidence}
            """)




if __name__ == '__main__':
    main()