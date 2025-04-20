import torch
from pathlib import Path
# import matplotlib.pyplot as plt

# Locate files relative to this script
BASE = Path(__file__).parent.resolve()
WEIGHTS = BASE / "best.pt"
IMG     = BASE / "santa.JPG"  

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
if not IMG.exists():
    raise FileNotFoundError(f"Image not found:   {IMG}")

# Load  YOLOv5 model
model = torch.hub.load(
    "ultralytics/yolov5",   
    "custom",               
    path=str(WEIGHTS),      
    force_reload=True       
)

results = model(str(IMG), augment=True)

# Display summary & image
results.print()   
results.show()  

# Print each detected class + confidence
print("\nDetected objects:")
for *box, conf, cls in results.xyxy[0].tolist():
    label = model.names[int(cls)]
    print(f" - {label:<10}  {conf*100:5.1f}%")
