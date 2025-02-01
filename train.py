import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from dataset import get_data_loaders
from models import EnsembleModel
import time
import logging
import gc

def train_model(args):
    # Setup logging
    logging.basicConfig(filename='training.log', level=logging.INFO,
                      format='%(asctime)s - %(message)s')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()  # Clear GPU cache

    # Get data loaders
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)

    # Create model
    model = EnsembleModel(num_models=args.num_models, num_classes=200).to(device)

    # Use mixed precision training
    scaler = torch.amp.GradScaler(device.type)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Use mixed precision training
                with torch.amp.autocast(device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Free up memory
                del outputs, loss
                torch.cuda.empty_cache()

                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                               f'Loss: {running_loss/(batch_idx+1):.4f}')
                    print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                          f'Loss: {running_loss/(batch_idx+1):.4f}')

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM error in batch {batch_idx}. Clearing cache and skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Free up memory
                del outputs
                torch.cuda.empty_cache()

        val_acc = 100. * correct / total

        # Log results
        logging.info(f'Epoch {epoch+1}/{args.epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Acc: {val_acc:.2f}%')
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')

        scheduler.step(val_acc)

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to CUB_200_2011 dataset')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_models', type=int, default=3,
                      help='Number of models in ensemble')

    args = parser.parse_args()
    train_model(args)
