import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os
import shutil
from argparse import ArgumentParser




def get_args():
    parser = ArgumentParser(description="Chương trình giám sát robot bằng YOLOv8 segmentation")
    parser.add_argument("--source", "-s", type=str, default=r"D:\Dataset\video_test2.mp4")
    parser.add_argument("--checkpoint", "-c", type=str, default=r"D:\Dataset\runs\segment\robot_gripper\weights\best.pt")
    parser.add_argument("--output", "-o", type=str, default=r"D:\Dataset\results\robot_monitor_output2.mp4")
    parser.add_argument("--scale", "-sc", type=float, default=0.6)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--threshold-stop", type=float, default=1.0, help="Stopped")
    parser.add_argument("--threshold-slow", type=float, default=4.0, help="Slow")
    parser.add_argument("--alert-delay", type=float, default=3.0, help="Stop for a long time")
    args = parser.parse_args()
    return args



def main():
    args = get_args()

    MODEL_PATH = args.checkpoint
    SOURCE = args.source
    OUTPUT_PATH = args.output

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    print(f"Run by : {'GPU' if device == 0 else 'CPU'}")



    colors = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (0, 0, 255),
        3: (255, 255, 0)
    }
    class_names = ['UR5 arm', 'Left gripper', 'Right gripper', 'Target']


    def get_robot_mask(result, shape):

        if result.masks is None:
            return np.zeros(shape[:2], np.uint8)
        mask_data = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        robot_mask = np.zeros(shape[:2], np.uint8)
        for i, mask in enumerate(mask_data):
            if classes[i] in [0, 1, 2]:
                mask = cv2.resize(mask, (shape[1], shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                robot_mask = np.maximum(robot_mask, mask)
        return robot_mask

    def calc_motion(prev_mask, curr_mask):

        diff = cv2.absdiff(prev_mask, curr_mask)
        return np.sum(diff) / curr_mask.size * 100



    THRESHOLD_STOP = args.threshold_stop
    THRESHOLD_SLOW = args.threshold_slow
    ALERT_DELAY = args.alert_delay


    cap = cv2.VideoCapture(int(SOURCE) if str(SOURCE).isdigit() else SOURCE)
    scale = args.scale
    font = cv2.FONT_HERSHEY_SIMPLEX


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


    prev_robot_mask = None
    motion_state = "unknown"
    state_start_time = None


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        results = model.predict(source=frame, device=device, conf=args.conf, verbose=False)

        for r in results:
            robot_mask = get_robot_mask(r, frame.shape)
            motion = calc_motion(prev_robot_mask, robot_mask) if prev_robot_mask is not None else 0


            current_state = "active"
            if motion < THRESHOLD_STOP:
                current_state = "stopped"
            elif motion < THRESHOLD_SLOW:
                current_state = "slow"

            if current_state != motion_state:
                motion_state = current_state
                state_start_time = time.time()

            elapsed = time.time() - state_start_time if state_start_time else 0


            if motion_state == "stopped" and elapsed >= ALERT_DELAY:
                status_text = f" Dung lau ({elapsed:.1f}s)"
                status_color = (0, 0, 255)
            elif motion_state == "slow" and elapsed >= ALERT_DELAY:
                status_text = f" Chuyen dong cham ({elapsed:.1f}s)"
                status_color = (0, 165, 255)
            elif motion_state == "stopped":
                status_text = f" Dung tam thoi ({elapsed:.1f}s)"
                status_color = (0, 100, 255)
            else:
                status_text = " Hoat dong binh thuong"
                status_color = (0, 255, 0)


            overlay = np.zeros_like(frame)
            if r.masks is not None:
                for i, mask in enumerate(r.masks.data.cpu().numpy()):
                    cls_id = int(r.boxes.cls.cpu().numpy()[i])
                    color = colors.get(cls_id, (255, 255, 255))
                    m = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    m = (m > 0.5).astype(np.uint8)
                    overlay[m == 1] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.45, 0)


            cv2.putText(frame, f"Ti le chuyen dong: {motion:.2f}%", (20, 40), font, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 75), font, 0.9, status_color, 2)
            cv2.putText(frame, f"Thoi gian trang thai: {elapsed:.1f}s", (20, 110), font, 0.6, (255, 255, 255), 1)

            prev_robot_mask = robot_mask.copy()


        cv2.imshow("Robot Arm Monitor", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"The resulting video is saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
