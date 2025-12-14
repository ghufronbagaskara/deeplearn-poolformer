"""
Script Eksperimen PoolFormer untuk CIFAR-100
Versi yang SUDAH DIPERBAIKI - Dijamin Jalan!
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from models import poolformer  # Import module poolformer

def main():
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
    # Batch size 32 agar ringan di laptop
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=0
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
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 1  # Cukup 1 epoch untuk bukti eksperimen
    print("=" * 60)
    print(f"Mulai Training selama {epochs} epoch...")
    print("=" * 60)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
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
            
            # Hitung akurasi per batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print status setiap 100 batch
            if i % 100 == 99:
                batch_acc = 100 * correct_train / total_train
                print(f'[Epoch {epoch + 1}, Batch {i + 1:4d}] Loss: {running_loss / 100:.4f} | Acc: {batch_acc:.2f}%')
                running_loss = 0.0
    
    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✓ Training Selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit)")
    print("=" * 60 + "\n")

    # --- 4. EVALUATION LOOP (UJIAN) ---
    print("=" * 60)
    print("Menghitung Akurasi Akhir pada Test Set...")
    print("=" * 60)
    
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

if __name__ == '__main__':
    main()
