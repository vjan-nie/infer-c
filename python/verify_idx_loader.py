from idx_loader import load_idx_images, load_idx_labels

images = load_idx_images("../data/raw/train-images-idx3-ubyte")
labels = load_idx_labels("../data/raw/train-labels-idx1-ubyte")

print("Images shape:", images.shape)   # expected: (60000, 28, 28)
print("Labels shape:", labels.shape)   # expected: (60000,)
print("First label:", labels[0])

for row in images[0]:
    print("".join("#" if px > 127 else "." for px in row))