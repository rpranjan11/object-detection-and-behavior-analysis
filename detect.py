from ultralytics import YOLO
import torch, sys

# Usage: python detect.py /path/to/video.mp4
src = sys.argv[1] if len(sys.argv) > 1 else "LiveCCTV_golden2.mp4"

# Choose a small, fast model first; we can scale up later
model = YOLO("yolov8n.pt")   # COCO pretrain (person, common animals, etc.)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# COCO class IDs for person + common animals
ANIMAL_CLASSES = [15,16,17,18,19,20,21,22,23]  # cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe
TARGETS = [0] + ANIMAL_CLASSES  # 0 = person

# Run inference and save annotated video
model.predict(
    source=src,
    device=device,
    imgsz=640,            # try 640 first; we can bump to 704/736 if needed
    conf=0.25,            # confidence threshold
    iou=0.45,             # NMS threshold
    classes=TARGETS,      # filter to people + animals
    agnostic_nms=True,
    vid_stride=1,         # set 2–3 to skip frames if you need more speed
    save=True,
    project="runs",
    name="detector_mps",
    show=False
)
print("Done. Check the runs/detector_mps/ folder for the annotated video.")
