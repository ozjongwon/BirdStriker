import torch
import argparse
from dataset import get_data_loaders
from models import EnsembleModel
from sklearn.metrics import classification_report
import numpy as np
import logging

def evaluate_model(args):
    # Setup logging
    logging.basicConfig(filename='evaluation.log', level=logging.INFO)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loader
    _, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    # Load model
    model = EnsembleModel(num_models=args.num_models, num_classes=200).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    report = classification_report(all_labels, all_preds)
    logging.info('Classification Report:')
    logging.info(report)

    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    logging.info(f'Overall Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to CUB_200_2011 dataset')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_models', type=int, default=3,
                      help='Number of models in ensemble')

    args = parser.parse_args()
    evaluate_model(args)
