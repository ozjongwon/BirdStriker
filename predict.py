# predict.py --model_path models\2025-02-02_09-30-07\best_model.pth --class_names class_names.json --image_path CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg


import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from models import EnsembleModel
import json
import logging
import os

class BirdPredictor:
    def __init__(self, model_path, class_names_file, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load class names
        with open(class_names_file, 'r') as f:
            self.class_names = json.load(f)

        # Create model
        self.model = EnsembleModel(num_models=3, num_classes=200).to(self.device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, top_k=5):
        """
        Predict bird species from image
        Args:
            image_path: Path to input image
            top_k: Number of top predictions to return
        Returns:
            List of (class_name, probability) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top k predictions
        top_prob, top_class = torch.topk(probabilities, top_k)

        # Convert to class names and probabilities
        predictions = []
        for i in range(top_k):
            class_idx = top_class[i].item()
            predictions.append({
                'class_name': self.class_names[str(class_idx)],
                'probability': float(top_prob[i].item())
            })

        return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict bird species from image')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--class_names', type=str, required=True,
                      help='Path to class names JSON file')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of top predictions to show')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize predictor
    predictor = BirdPredictor(args.model_path, args.class_names)

    # Make prediction
    predictions = predictor.predict(args.image_path, args.top_k)

    # Print results
    print(f"\nPredictions for {os.path.basename(args.image_path)}:")
    print("-" * 50)
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['class_name']}: {pred['probability']:.4f}")

if __name__ == '__main__':
    main()
