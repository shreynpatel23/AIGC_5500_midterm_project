import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import itertools
import time
import pandas as pd

def train_adam(model, trainset, kf, hyperparams, batch_size):
    """
    Trains the given model using the Adam optimizer with a hyperparameter search.
    
    Parameters:
    model (torch.nn.Module): The neural network model to be trained.
    trainset (Dataset): The dataset used for training.
    kf (KFold): K-Fold cross-validator for splitting the dataset.
    hyperparams (dict): Dictionary containing hyperparameters like learning rates, betas, weight decay, and epochs.
    batch_size (int): Batch size for training.
    """
    
    # Select appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    # Extract hyperparameter search space from input dictionary
    learning_rates = hyperparams['learning_rates']
    betas = hyperparams['betas']
    weight_decay = hyperparams['weight_decays']
    epochs = hyperparams['epochs']

    # Initialize tracking variables for best results
    best_acc = 0
    best_params = {}
    
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    
    results = []
    
    # Iterate over all possible hyperparameter combinations
    for lr, beta, decay in itertools.product(learning_rates, betas, weight_decay):
        fold_accs = [] # Store accuracy for each fold
        total_time = 0 # Track total training time
        writer = SummaryWriter(log_dir=f'logs/adam_lr{lr}_beta{beta}_weight_decay{decay}')
        
        # K-Fold cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
            train_loader, val_loader = get_data_loaders(trainset, train_idx, val_idx, batch_size)

            optimizer = optim.Adam(model.parameters(), lr=lr, betas=beta, weight_decay=decay)

            print(f"\nTraining Fold {fold+1} with LR={lr}, Beta={beta}, Weight Decay={decay}")
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                train_correct = 0
                total_train = 0
                start_time = time.time()
                
                for images, labels in train_loader:
                    # Move data to GPU.
                    images, labels = images.to(device), labels.to(device)
                    
                    # Zero the parameter gradients.
                    optimizer.zero_grad()
                    
                    # Forward pass.
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    train_loss += loss.item()
                    
                    # prediction
                    _, indices = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    train_correct += (indices == labels).sum().item()
                    
                    # Backward pass and update.
                    loss.backward()
                    optimizer.step()

                avg_train_loss = train_loss / len(train_loader)
                avg_train_acc = 100 * train_correct / total_train
                
                train_losses.append(avg_train_loss)
                train_accs.append(avg_train_acc)
                
                # Add to TensorBoard
                writer.add_scalar(f"Train Loss Per Epoch", avg_train_loss, epoch + fold * epochs)
                writer.add_scalar(f"Train Accuracy Per Epoch", avg_train_acc, epoch + fold * epochs)

                # Validation
                model.eval()
                test_loss = 0
                test_correct = 0
                total_test = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        # Move data to GPU.
                        images, labels = images.to(device), labels.to(device)
                        
                        
                        # Forward pass.
                        outputs = model(images)
                        loss = loss_function(outputs, labels)
                        test_loss += loss.item()
                        
                        _, predicted = torch.max(outputs, 1)
                        total_test += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                
                avg_test_loss = test_loss / len(val_loader)
                avg_test_acc = 100 * test_correct / total_test
                
                
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                fold_accs.append(avg_test_acc)
                
                # Add to TensorBoard
                writer.add_scalar(f"Validation Loss Per Epoch", avg_test_loss, epoch + fold * epochs)
                writer.add_scalar(f"Validation Accuracy Per Epoch", avg_test_acc, epoch + fold * epochs)
                
                time_taken = time.time() - start_time
                total_time += time_taken

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.2f}%, "
                      f"Test Loss={avg_test_loss:.4f}, Test Acc={avg_test_acc:.2f}%, Time={time_taken:.2f}s")
        
        print(f"fold_accs: {fold_accs}")
        avg_fold_acc = sum(fold_accs) / len(fold_accs)
        print(f"Average Fold Accuracy: {avg_fold_acc:.2f}%")
        # Store the best performing hyperparameters
        if avg_fold_acc > best_acc:
            print(f"previous best accuracy {best_acc:.2f}%")
            print(f"previous best parameters: {best_params}")
            best_acc = avg_fold_acc
            best_params = {'learning_rate': lr, 'betas': beta, 'weight_decay': decay}
            print(f"new best validation Accuracy: {best_acc:.2f}%")
            print(f"new best parameters: {best_params}")
            
        # Save results for this hyperparameter combination
        results.append({
            "learning_rate": lr,
            "betas": beta,
            "weight_decay": decay,
            "avg_fold_acc": avg_fold_acc,
            "total_time (s)": total_time
        })
        
        # Close the TensorBoard writer
        writer.close()
    
    print("\nBest Hyperparameters for Adam:", best_params)
    print(f"Best Accuracy: {best_acc:.2f}%")
    
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    # Optionally, export to CSV
    results_df.to_csv("adam_training_results.csv", index=False)
    
