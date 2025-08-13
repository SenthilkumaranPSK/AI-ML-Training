import argparse
import cv2
from ultralytics import YOLO
import time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 *.pt model")
    p.add_argument("--device", type=str, default="", help="CUDA device like '0' or '0,1' or 'cpu' (auto if empty)")
    p.add_argument("--source", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--show-fps", action="store_true", help="Overlay FPS counter")
    return p.parse_args()

def main():
    args = parse_args() 
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.source}")
    prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Inference (BGR numpy array is acceptable)
        results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)
        annotated = results[0].plot()  # BGR image with boxes & labels
        if args.show_fps:
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLOv8 Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
