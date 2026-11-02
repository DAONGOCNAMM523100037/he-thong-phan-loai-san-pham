
from pathlib import Path
import random
import shutil

# Thư mục gốc của dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Nơi chứa 4 nhóm ảnh gốc
SOURCE_DIR = PROJECT_DIR / "dataset_goc"

# Nơi lưu dữ liệu sau khi chia
OUTPUT_DIR = PROJECT_DIR / "data_chia"

# Tên 4 nhóm sản phẩm
CLASSES = [
    "nuoc_uong",
    "ca_phe",
    "thuc_an",
    "mi_an_lien"
]

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Giúp kết quả chia có thể tái lập
random.seed(42)

# Các định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def main():
    # Kiểm tra thư mục ảnh gốc
    if not SOURCE_DIR.exists():
        print(f"Khong tim thay thu muc: {SOURCE_DIR}")
        return

    # Không ghi đè dữ liệu đã chia trước đó
    if OUTPUT_DIR.exists():
        print(f"Thu muc {OUTPUT_DIR.name} da ton tai.")
        print("Hay doi ten hoac sao luu thu muc nay truoc khi chay lai.")
        return

    # Kiểm tra ảnh của từng nhóm
    image_groups = {}

    for class_name in CLASSES:
        class_dir = SOURCE_DIR / class_name

        if not class_dir.exists():
            print(f"Thieu thu muc: {class_dir}")
            return

        images = [
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < 3:
            print(
                f"Nhom {class_name} chi co {len(images)} anh. "
                "Can it nhat 3 anh de chia du lieu."
            )
            return

        image_groups[class_name] = images

    print("Bat dau chia du lieu...\n")

    # Tạo thư mục train, val, test
    for split_name in ["train", "val", "test"]:
        for class_name in CLASSES:
            (OUTPUT_DIR / split_name / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    # Chia ảnh riêng cho từng nhóm
    for class_name, images in image_groups.items():
        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        split_groups = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, selected_images in split_groups.items():
            destination = OUTPUT_DIR / split_name / class_name

            for image_path in selected_images:
                shutil.copy2(image_path, destination / image_path.name)

        print(f"Nhom: {class_name}")
        print(f"  Tong so anh: {total}")
        print(f"  Train:       {len(train_images)}")
        print(f"  Validation:  {len(val_images)}")
        print(f"  Test:        {len(test_images)}")
        print()

    print("DA CHIA DU LIEU THANH CONG!")
    print(f"Du lieu moi duoc luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
