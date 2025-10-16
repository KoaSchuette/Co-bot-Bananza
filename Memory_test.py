import cv2
from ultralytics import YOLO

Conf_Threshold = 0.7

model = YOLO("640Best.pt")
cap = cv2.VideoCapture(0)

class Card_back:
    ID
    poly
    prev_poly
    value

def Distance_Detect(poly, prev_poly):

    pass




while True:
    ret, frame = cap.read()
    if not ret:
        print("lost signal to camera")
        break
    results = model(frame)
    r = results[0]
    annotated_frame = frame.copy()
    for i in range(len(r.obb.conf)):
        conf = float(r.obb.conf[i].item())
        if conf < Conf_Threshold:
            continue
        
        cls_id = int(r.obb.cls[i].item())
        label = model.names.get(cls_id, str(cls_id))

        # polygon points (x1,y1, x2,y2, x3,y3, x4,y4)
        poly = r.obb.xyxyxyxy[i].cpu().numpy().astype(int).reshape(4, 2)


























        color = (0, 255, 0)
        cv2.polylines(annotated_frame, [poly], isClosed=True, color=color, thickness=2)

        # put label near the top-left corner of the polygon
        x_min, y_min = poly[:,0].min(), poly[:,1].min()
        text = f"{label} {conf:.2f} {i}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x_min, y_min - th - 6), (x_min + tw + 6, y_min), color, -1)
        cv2.putText(annotated_frame, text, (x_min + 3, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        print(f"{label} {conf:.2f} poly={poly.tolist()}")
        drew_any = True


    cv2.imshow("ryan is so beautiful", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

