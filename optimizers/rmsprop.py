import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import itertools
import time
import pandas as pd

def train_rmsprop(model, trainset, kf, hyperparams, batch_size):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = nn.CrossEntropyLoss()# Define loss function for classification

    # Define hyperparameter value
    learning_rates = hyperparams['learning_rates']
    alphas = hyperparams['alphas']
    epochs = hyperparams['epochs']

    best_acc = 0  # Track the best accuracy
    best_params = {} # Store the best hyperparameters
    
    train_accs = []  # List to store training accuracies
    test_accs = []  # List to store validation accuracies
    train_losses = []  # List to store training losses
    test_losses = []  # List to store validation losses
    
    results = [] # List to store results of each hyperparameters combination

    # Iterate over all combinations of learning rates and alpha values
    for lr, alpha in itertools.product(learning_rates, alphas):
         # Store accuracy for each fold
        fold_accs = []
         # Track total training time for this setting
        total_time = 0
        writer = SummaryWriter(log_dir=f'logs/rms_prop_lr{lr}_alpha{alpha}')
        #  k-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
            train_loader, val_loader = get_data_loaders(trainset, train_idx, val_idx, batch_size)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha) # Initialize optimizer

            print(f"\nTraining Fold {fold+1} with LR={lr}, Alpha={alpha}")
            
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
                    optimizer.step() # Update weights
                    
                 # Compute average training loss and accuracy
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
                        # Move data to Device.
                        images, labels = images.to(device), labels.to(device)
                        
                        
                        # Forward pass.
                        outputs = model(images)
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
                
                # Add to TensorBoard
                writer.add_scalar(f"Validation Loss Per Epoch", avg_test_loss, epoch + fold * epochs)
                writer.add_scalar(f"Validation Accuracy Per Epoch", avg_test_acc, epoch + fold * epochs)
                
                time_taken = time.time() - start_time
                total_time += time_taken

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.2f}%, "
                      f"Test Loss={avg_test_loss:.4f}, Test Acc={avg_test_acc:.2f}%, Time={time_taken:.2f}s")
            
        avg_fold_acc = sum(fold_accs) / len(fold_accs)
        # Store the best performing hyperparameters
        if avg_fold_acc > best_acc:
            best_acc = avg_fold_acc
            best_params = {'learning_rate': lr, 'alpha': alpha}

            
        # Save results for this hyperparameter combination
        results.append({
            "learning_rate": lr,
            "alpha": alpha,
            "avg_fold_acc": avg_fold_acc,
            "total_time (s)": total_time
        })

        writer.close() # Close TensorBoard writer
    
    print("\nBest Hyperparameters for RMSProp:", best_params)
    print(f"Best Accuracy: {best_acc:.2f}%")
    
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    # Optionally, export to CSV
    results_df.to_csv("rmsprop_results.csv", index=False)
