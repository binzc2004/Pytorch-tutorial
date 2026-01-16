import os
import random
import shutil
from pathlib import Path


def split_dataset(
    src_dir="dog_cat",
    dst_dir="data",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于 1"

    classes = [d.name for d in src_dir.iterdir() if d.is_dir()]
    print("发现类别：", classes)

    # 创建目标目录
    for split in ["train", "val", "test"]:
        for cls in classes:
            (dst_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # 对每个类别分别划分
    for cls in classes:
        cls_dir = src_dir / cls
        images = list(cls_dir.iterdir())
        images = [img for img in images if img.suffix.lower() in [".jpg", ".png", ".jpeg"]]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        print(f"{cls}: total={total}, "
              f"train={len(train_imgs)}, "
              f"val={len(val_imgs)}, "
              f"test={len(test_imgs)}")

        def copy_images(img_list, split):
            for img in img_list:
                shutil.copy(
                    img,
                    dst_dir / split / cls / img.name
                )

        copy_images(train_imgs, "train")
        copy_images(val_imgs, "val")
        copy_images(test_imgs, "test")

    print("✅ 数据集划分完成！")


if __name__ == "__main__":
    split_dataset(
        src_dir="cat_dog",
        dst_dir="data"
    )
