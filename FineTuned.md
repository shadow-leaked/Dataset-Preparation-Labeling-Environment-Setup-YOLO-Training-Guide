# 📘 Fine-Tuning a YOLO Model With New Images (Complete Guide)

This guide explains how to continue training your already-trained YOLO model (e.g., `best.pt`) by adding new images and improving accuracy over time.  
This section is formatted so you can directly place it inside a repository documentation file.

---

## ✅ 1. Understand What Fine-Tuning Means

Fine-tuning means:

**Reuse your previous trained model (`best.pt`) + add new images + train again.**

This avoids training from scratch and helps the model learn difficult real-world cases.

You should fine-tune when:

- You added new images  
- You captured new angles, lighting, distances  
- The model performs poorly in certain scenarios  
- The object has variations your original dataset did not include  

---

## 📂 2. Add New Images to Your Dataset

Your dataset folder **must** follow this structure:

│── train/ \
│ ├── images/ \
│ └── labels/ \
│── valid/ \
│ ├── images/ \
│ └── labels/ 


### Steps:

1. Capture new images  
2. Import them into **Label Studio**  
3. Label the new images in **YOLO format**  
4. Export the labeled dataset as a **YOLO ZIP**  
5. Unzip and copy:
    - images → dataset/train/images/
    - labels → dataset/train/labels/

### ⚠️ Important  
Filenames must match exactly:

> image_51.jpg → image_51.txt

---

## 🧪 3. Validate the Dataset Again (Recommended)

Run your dataset validation script:

```bash
python check_dataset.py
```
Check carefully for:
- Images without labels
- Labels without images
- Wrong or mismatched filenames
- Extra files (e.g., classes.txt — delete it)

A clean dataset is critical before fine-tuning.

## 🗂️ 4. Keep the Same data.yaml

You do not need a new YAML file if:
- The classes are the same
- The dataset folder structure is unchanged

A typical data.yaml looks like:
```bash
train: dataset/train/images
val: dataset/valid/images

nc: 1
names: ["object"]
```
> If you added new classes → update nc and names accordingly.

## 🧠 5. Fine-Tuning Using Previous Weights

This is the core step of the process.

> ❌ Wrong (training from scratch)
```bash
model=yolov8n.pt
```
> ✅ Correct (fine-tuning)

Use your previously trained model:
```bash
model="runs/detect/train/weights/best.pt"
```
### Run this command to fine-tune:
```bash
yolo detect train data="dataset/data.yaml" model="runs/detect/train/weights/best.pt" epochs=30 imgsz=640 batch=8 device=0 resume=False
```

## 🎛️ 6. How Many Epochs Should You Use?

| Added Images         | Recommended Epochs |
|----------------------|--------------------|
| 20–80 new images     | 20–30 epochs       |
| 100–200 new images   | 30–50 epochs       |
| 300+ new images      | 60–100 epochs      |
| New huge dataset     | 100–150 epochs     |

Why fewer epochs?  
Because we're continuing training, not starting from zero.

---

## 🧩 7. YOLO Creates a New Training Folder

After fine-tuning, YOLO automatically generates new training directories:

```bash
runs/detect/train2/
runs/detect/train3/
runs/detect/train4/
```
 Your updated model will be located here:
```bash
runs/detect/trainX/weights/best.pt
```
This becomes your new primary model.

## 🎥 8. Use New Weights in Real-Time Detection

Update your inference script:

```python
model = YOLO("runs/detect/train2/weights/best.pt")
```
Run your detection program:

```bash
python realtime_detection.py
```
## 🎯 9. Best Practices for Fine-Tuning
#### ✔ Add Challenging Images

Your model improves the most when you add:
- Low-light images
- Blurry frames
- Occlusions
- Long-distance shots
- Reflective / complex backgrounds
- False-positive examples
- Drone movement frames

### ✔ Maintain a 70/30 Split

> Keep validation clean — don’t move everything into training.

### ✔ Use Augmentations

> YOLO automatically applies augmentations, which is extremely useful for smaller datasets.

### ✔ Train in Multiple Rounds

> Real-world model quality improves across several fine-tuning cycles.

## 🧪 10. When Should You Re-Train From Zero?

Retraining from scratch is required only when:
- You changed the class structure
- The dataset becomes extremely large (1,000+ images)
- You switch YOLO versions (e.g., YOLOv8 → YOLOv11)

Otherwise, fine-tuning is sufficient.

## 🏁 Summary

### Fine-tuning workflow:

1. Add new images
2. Label them
3. Place them in train/images + train/labels
4. Validate dataset
5. Use best.pt as the base model
6. Train again for ~20–50 epochs
7. Use the new weights from trainX/weights/best.pt

This produces a stronger, more robust, and more accurate model compared to training from scratch.
