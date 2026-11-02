
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# 1. CAU HINH DU AN
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data_chia"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

CLASS_NAMES = [
    "nuoc_uong",
    "ca_phe",
    "thuc_an",
    "mi_an_lien"
]

tf.keras.utils.set_random_seed(SEED)


# =====================================================
# 2. KIEM TRA DU LIEU
# =====================================================

print("\n========== KIEM TRA DU LIEU ==========")

for split in ["train", "val", "test"]:
    for class_name in CLASS_NAMES:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc: {folder}"
            )

print("Da tim thay day du cac thu muc du lieu!")


# =====================================================
# 3. DOC DU LIEU ANH
# =====================================================

print("\n========== DOC DU LIEU ANH ==========")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nThu tu nhan:")
for index, name in enumerate(CLASS_NAMES):
    print(f"{index}: {name}")


# =====================================================
# 4. TANG CUONG DU LIEU
# =====================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")


# =====================================================
# 5. XAY DUNG MO HINH CNN MOBILENETV2
# =====================================================

print("\n========== XAY DUNG MO HINH ==========")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Dong bang cac lop da duoc huan luyen truoc
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

# Tang cuong du lieu
x = data_augmentation(inputs)

# Chuyen gia tri pixel tu 0-255 sang -1 den 1
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

# Trich xuat dac trung bang MobileNetV2
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================================================
# 6. CAC CALLBACK
# =====================================================

best_model_path = MODEL_DIR / "best_model.keras"

callbacks = [
    ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]


# =====================================================
# 7. HUAN LUYEN MO HINH
# =====================================================

print("\n========== BAT DAU HUAN LUYEN ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =====================================================
# 8. LUU MO HINH TOT NHAT
# =====================================================

print("\n========== LUU MO HINH ==========")

best_model = tf.keras.models.load_model(
    best_model_path
)

final_model_path = MODEL_DIR / "product_classifier.keras"

best_model.save(final_model_path)

print(f"Da luu mo hinh: {final_model_path}")


# Luu thu tu cac nhan
labels_path = MODEL_DIR / "class_names.json"

with open(labels_path, "w", encoding="utf-8") as file:
    json.dump(
        CLASS_NAMES,
        file,
        ensure_ascii=False,
        indent=4
    )

print(f"Da luu danh sach nhan: {labels_path}")


# =====================================================
# 9. DANH GIA TREN TAP TEST
# =====================================================

print("\n========== DANH GIA MO HINH ==========")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Lay nhan that va ket qua du doan
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

predictions = best_model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)


# Bao cao phan loai
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nBAO CAO PHAN LOAI:")
print(report)

report_path = OUTPUT_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)
    file.write(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

print(f"Da luu bao cao: {report_path}")


# =====================================================
# 10. VE BIEU DO HUAN LUYEN
# =====================================================

print("\n========== VE BIEU DO ==========")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

chart_path = CHART_DIR / "training_history.png"
plt.savefig(chart_path, dpi=300)
plt.close()


# =====================================================
# 11. VE MA TRAN NHAM LAN
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

plt.figure(figsize=(8, 6))

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Nhan du doan")
plt.ylabel("Nhan thuc te")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()

cm_path = CHART_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()


# =====================================================
# 12. HOAN TAT
# =====================================================

print("\n========================================")
print("      HUAN LUYEN MO HINH HOAN TAT")
print("========================================")

print(f"Do chinh xac Test: {test_accuracy * 100:.2f}%")
print(f"Mo hinh: {final_model_path}")
print(f"Bieu do: {chart_path}")
print(f"Ma tran nham lan: {cm_path}")
print(f"Bao cao: {report_path}")

print("\nBON NHOM SAN PHAM:")
for name in CLASS_NAMES:
    print("-", name)
