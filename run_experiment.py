"""
Script Eksperimen PoolFormer untuk CIFAR-100
Versi yang SUDAH DIPERBAIKI - Dijamin Jalan!
+ Checkpoint Saving & Loading untuk Resume Training
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import json
import argparse
from models import poolformer  # Import module poolformer

# --- KONFIGURASI CHECKPOINT ---
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'poolformer_cifar100_latest.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'poolformer_cifar100_best.pth')
METRICS_PATH = os.path.join(CHECKPOINT_DIR, 'training_metrics.json')

def parse_args():
    parser = argparse.ArgumentParser(description='PoolFormer CIFAR-100 Training')
    parser.add_argument('--epochs', type=int, default=10, help='Jumlah epoch untuk training (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--resume', action='store_true', help='Resume dari checkpoint terakhir')
    parser.add_argument('--reset', action='store_true', help='Hapus checkpoint dan training dari awal')
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, best_acc, metrics_history, is_best=False):
    """Simpan checkpoint training"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'metrics_history': metrics_history
    }
    
    # Simpan checkpoint terbaru
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"💾 Checkpoint disimpan: {CHECKPOINT_PATH}")
    
    # Simpan model terbaik jika ini adalah yang terbaik
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"🏆 Best model disimpan: {BEST_MODEL_PATH}")
    
    # Simpan metrics ke JSON (untuk plotting/analisis)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"📊 Metrics disimpan: {METRICS_PATH}")

def load_checkpoint(model, optimizer, device):
    """Load checkpoint jika ada"""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📂 Loading checkpoint dari: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        metrics_history = checkpoint.get('metrics_history', {'train_loss': [], 'train_acc': [], 'test_acc': []})
        print(f"✓ Checkpoint loaded! Melanjutkan dari epoch {start_epoch}, Best Acc: {best_acc:.2f}%")
        return start_epoch, best_acc, metrics_history
    return 0, 0.0, {'train_loss': [], 'train_acc': [], 'test_acc': []}

def main():
    # Parse arguments
    args = parse_args()
    
    # Reset checkpoint jika diminta
    if args.reset:
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            print("🗑️ Checkpoint dihapus, training dari awal...")
    
    # --- 1. SETUP DATASET (CIFAR-100) ---
    print("=" * 60)
    print("Mendownload & Menyiapkan Data CIFAR-100...")
    print("=" * 60)
    
    # Transformasi agar gambar lebih bervariasi (Augmentasi Sederhana)
    transform_train = transforms.Compose([
        transforms.Resize(224),  # PoolFormer membutuhkan input 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"✓ Dataset siap: {len(trainset)} training, {len(testset)} testing\n")

    # --- 2. SETUP MODEL ---
    print("=" * 60)
    print("Membangun Model PoolFormer-S12...")
    print("=" * 60)
    
    # Load model
    model = poolformer.poolformer_s12(pretrained=False)
    
    # Ubah head untuk 100 kelas CIFAR-100
    model.head = nn.Linear(model.head.in_features, 100)  # Ubah output ke 100 kelas  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✓ Model berhasil dimuat")
    print(f"✓ Menggunakan Device: {device}")
    
    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total Parameters: {total_params:,}\n")

    # --- 3. TRAINING LOOP (BELAJAR) ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load checkpoint jika ada (untuk resume training)
    start_epoch, best_acc, metrics_history = load_checkpoint(model, optimizer, device)
    
    epochs = args.epochs  # Total epoch dari argument
    print("=" * 60)
    print(f"📋 Konfigurasi: epochs={epochs}, batch_size={args.batch_size}, lr={args.lr}")
    if start_epoch > 0:
        print(f"Melanjutkan Training dari epoch {start_epoch + 1} sampai {epochs}...")
    else:
        print(f"Mulai Training selama {epochs} epoch...")
    print("=" * 60)
    
    # Skip jika sudah selesai training
    if start_epoch >= epochs:
        print(f"✓ Training sudah selesai sebelumnya ({epochs} epoch)")
        print(f"✓ Best Accuracy: {best_acc:.2f}%")
        print("💡 Gunakan --reset untuk training ulang dari awal, atau --epochs N untuk menambah epoch")
    else:
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                
                # Hitung akurasi per batch
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Print status setiap 100 batch
                if i % 100 == 99:
                    batch_acc = 100 * correct_train / total_train
                    print(f'[Epoch {epoch + 1}, Batch {i + 1:4d}] Loss: {running_loss / 100:.4f} | Acc: {batch_acc:.2f}%')
                    running_loss = 0.0
            
            # Hitung metrics per epoch
            train_acc = 100 * correct_train / total_train
            avg_loss = epoch_loss / len(trainloader)
            
            # Evaluasi test accuracy setiap epoch
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            test_acc = 100 * correct_test / total_test
            
            # Simpan metrics
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['test_acc'].append(test_acc)
            
            # Cek apakah ini model terbaik
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
            
            print(f"\n📈 Epoch {epoch + 1} Summary: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")
            
            # Simpan checkpoint setiap epoch
            save_checkpoint(model, optimizer, epoch, best_acc, metrics_history, is_best)
            print()
        
        training_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✓ Training Selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit)")
        print("=" * 60 + "\n")

    # --- 4. EVALUATION LOOP (UJIAN) - Load Best Model ---
    print("=" * 60)
    print("Menghitung Akurasi Akhir pada Test Set (dengan Best Model)...")
    print("=" * 60)
    
    # Load best model untuk evaluasi final
    if os.path.exists(BEST_MODEL_PATH):
        best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"✓ Loaded best model dari: {BEST_MODEL_PATH}")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total
    
    print("\n" + "=" * 60)
    print(f"╔{'═' * 58}╗")
    print(f"║  FINAL ACCURACY: {final_accuracy:5.2f}%{' ' * 35}║")
    print(f"╚{'═' * 58}╝")
    print("=" * 60)
    print("\n📸 SCREENSHOT SEKARANG untuk presentasi Anda!")
    print("=" * 60)
    
    # --- 5. INFO CHECKPOINT ---
    print("\n💡 INFORMASI CHECKPOINT:")
    print(f"   - Checkpoint terbaru: {CHECKPOINT_PATH}")
    print(f"   - Model terbaik: {BEST_MODEL_PATH}")
    print(f"   - Training metrics: {METRICS_PATH}")
    print(f"   - Untuk training ulang dari awal, hapus folder 'checkpoints/'")
    print("=" * 60)

if __name__ == '__main__':
    main()
