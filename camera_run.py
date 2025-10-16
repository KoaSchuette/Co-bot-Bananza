import cv2
from ultralytics import YOLO
import time
import torch

CONF_THRESHOLD = 0.8

# Force CUDA and optimize model
model = YOLO("640Best.pt")
model.to('cuda')  # Ensure model is on GPU
torch.cuda.empty_cache()  # Clear GPU memory

# Warm up the model (important for performance)
print("Warming up model...")
dummy_frame = torch.zeros((1, 3, 960, 960)).cuda()
for _ in range(3):
    _ = model.predict(dummy_frame, verbose=False)
print("Model ready!")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # buffer to get latest frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Resolution Request
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 30)  # FPS Request

fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("couldn't read frame(camera Unplugged?)")
        break
    results = model.track(frame, verbose=False, conf=CONF_THRESHOLD, device='cuda', persist=True, tracker="bytetrack.yaml")
    r = results[0]
    annotated_frame = frame.copy()
    if r.obb is not None and len(r.obb.conf) > 0:
        for i in range(len(r.obb.conf)):
            conf = float(r.obb.conf[i].item())
            cls_id = int(r.obb.cls[i].item())
            label = model.names.get(int(r.obb.cls[i].item()), str(cls_id))
            track_id = int(r.obb.id[i].item()) if r.obb.id is not None else 0

            # polygon format (x1,y1, x2,y2, x3,y3, x4,y4)
            poly = r.obb.xyxyxyxy[i].cpu().numpy().astype(int).reshape(4, 2)
            color = (0, 255, 0)
            cv2.polylines(annotated_frame, [poly], isClosed=True, color=color, thickness=2)
            # Red Dot thing
            center_x = int(poly[:, 0].mean())
            center_y = int(poly[:, 1].mean())
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
            
            # Print center point
            print(f"ID:{track_id} {label} - Center: ({center_x}, {center_y})")
            
            x_min, y_min = poly[:,0].min(), poly[:,1].min()
            text = f"ID:{track_id} {label} {conf:.2f}"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x_min, y_min - th - 6), (x_min + tw + 6, y_min), color, -1)
            cv2.putText(annotated_frame, text, (x_min + 3, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0: 
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("RYAN", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()  # Clean up GPU memory
