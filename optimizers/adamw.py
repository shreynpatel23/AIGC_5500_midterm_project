import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import itertools
import time
import pandas as pd

def train_adamw(model, trainset, kf, hyperparams, batch_size):
    """Trains the model using AdamW optimizer with hyperparameter tuning and K-fold cross-validation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    loss_function = nn.CrossEntropyLoss()  # Loss function for classification

    # Extract hyperparameter ranges
    learning_rates = hyperparams['learning_rates']
    betas = hyperparams['betas']
    weight_decay = hyperparams['weight_decays']
    epochs = hyperparams['epochs']

    best_acc = 0  # Track the best accuracy
    best_params = {}  # Store best hyperparameters
    
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    results = []  # Store results for each hyperparameter combination

    # Iterate over all hyperparameter combinations
    for lr, beta, decay in itertools.product(learning_rates, betas, weight_decay):
        fold_accs = []  # Store accuracy for each fold
        total_time = 0  # Track total training time
        writer = SummaryWriter(log_dir=f'logs/adam_w_lr{lr}_beta{beta}_weight_decay{decay}')  # TensorBoard logging

        # K-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
            train_loader, val_loader = get_data_loaders(trainset, train_idx, val_idx, batch_size)
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=beta, weight_decay=decay)

            print(f"\nTraining Fold {fold+1} with LR={lr}, Beta={beta}, Weight Decay={decay}")
            
            # Train for the specified number of epochs
            for epoch in range(epochs):
                model.train()
                train_loss, train_correct, total_train = 0, 0, 0
                start_time = time.time()
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)  # Move data to GPU
                    optimizer.zero_grad()  # Reset gradients
                    
                    outputs = model(images)  # Forward pass
                    loss = loss_function(outputs, labels)
                    train_loss += loss.item()
                    
                    _, indices = torch.max(outputs.data, 1)  # Get predictions
                    total_train += labels.size(0)
                    train_correct += (indices == labels).sum().item()
                    
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update weights

                # Compute average training loss and accuracy
                avg_train_loss = train_loss / len(train_loader)
                avg_train_acc = 100 * train_correct / total_train
                
                train_losses.append(avg_train_loss)
                train_accs.append(avg_train_acc)
                
                # Log to TensorBoard
                writer.add_scalar("Train Loss Per Epoch", avg_train_loss, epoch + fold * epochs)
                writer.add_scalar("Train Accuracy Per Epoch", avg_train_acc, epoch + fold * epochs)

                # Validation phase
                model.eval()
                test_loss, test_correct, total_test = 0, 0, 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        
                        outputs = model(images)  # Forward pass
                        loss = loss_function(outputs, labels)
                        test_loss += loss.item()
                        
                        _, predicted = torch.max(outputs, 1)
                        total_test += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                
                # Compute average validation loss and accuracy
                avg_test_loss = test_loss / len(val_loader)
                avg_test_acc = 100 * test_correct / total_test
                
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                fold_accs.append(avg_test_acc)
                
                # Log to TensorBoard
                writer.add_scalar("Validation Loss Per Epoch", avg_test_loss, epoch + fold * epochs)
                writer.add_scalar("Validation Accuracy Per Epoch", avg_test_acc, epoch + fold * epochs)

                time_taken = time.time() - start_time  # Track time per epoch
                total_time += time_taken

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.2f}%, "
                      f"Test Loss={avg_test_loss:.4f}, Test Acc={avg_test_acc:.2f}%, Time={time_taken:.2f}s")

        # Compute average accuracy across folds
        avg_fold_acc = sum(fold_accs) / len(fold_accs)
        print(f"Average Fold Accuracy: {avg_fold_acc:.2f}%")

        # Store best performing hyperparameters
        if avg_fold_acc > best_acc:
            print(f"Previous Best Accuracy: {best_acc:.2f}%")
            print(f"Previous Best Parameters: {best_params}")
            best_acc = avg_fold_acc
            best_params = {'learning_rate': lr, 'betas': beta, 'weight_decay': decay}
            print(f"New Best Accuracy: {best_acc:.2f}%")
            print(f"New Best Parameters: {best_params}")
            
        # Save results for this hyperparameter combination
        results.append({
            "learning_rate": lr,
            "betas": beta,
            "weight_decay": decay,
            "avg_fold_acc": avg_fold_acc,
            "total_time (s)": total_time
        })

        writer.close()  # Close TensorBoard writer
    
    print("\nBest Hyperparameters for AdamW:", best_params)
    print(f"Best Accuracy: {best_acc:.2f}%")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("adam_w_training_results.csv", index=False)
