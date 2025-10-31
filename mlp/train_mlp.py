import sys, os, time, torch, torch.nn as nn, torch.optim as optim

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # Prevent multiprocessing issues on Windows

    # Path setup
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_mlp import MLP
    from data_loader import train_loader, val_loader, test_loader
    from utils import plot_curves

    # Environment configuration
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model, loss, optimizer and scheduler
    model = MLP(hidden1=128, hidden2=64, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    # Logs
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    EPOCHS = 10
    best_val_acc = 0.0

    # Define a fixed save path (always in the same folder as this script)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp_best.pth")

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        correct, total, loss_sum = 0, 0, 0

        # Training phase
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} processed")

        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(correct / total)

        # Validation phase
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / val_total)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} "
              f"| Time: {time.time() - start:.1f}s")

        # Save the best model (fixed save path)
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated at epoch {epoch+1} (Val Acc: {best_val_acc:.4f})")

    print("\nEvaluating best saved model on test set...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        torch.save(model.state_dict(), save_path)
        print("No previous model found — current model saved as mlp_best.pth")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Always save final model to script directory
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved at: {save_path}")

    plot_curves(train_accs, val_accs, train_losses, val_losses)
