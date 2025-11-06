import torch
from ultralytics import YOLO
import os
from argparse import ArgumentParser
from tqdm import tqdm
import time


def get_args():
    parser = ArgumentParser(description="Training robot segmentation model")

    parser.add_argument("--model", "-m", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--data", "-d", type=str, default=os.path.join(r"D:\Dataset\Robot_gripper\data.yaml"))

    parser.add_argument("--epochs", "-e", type=int, default=30, help="number of epochs")
    parser.add_argument("--imgsz", "-i", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--batch", "-b", type=int, default=2, help="Batch size")
    parser.add_argument("--workers", "-w", type=int, default=0)
    parser.add_argument("--device", "-dev", type=int, default=0)
    parser.add_argument("--optimizer", "-opt", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", type=str, default="ram")

    parser.add_argument("--project", "-p", type=str, default="runs/segment")
    parser.add_argument("--name", "-n", type=str, default="robot_gripper")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = YOLO(args.model)
    start_time = time.time()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tqdm(total=1, desc="Training YOLOv8", ncols=100, colour="cyan") as pbar:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            amp=args.amp,
            cache=args.cache,
            optimizer=args.optimizer,
            lr0=args.lr0,
            patience=args.patience,
            project=args.project,
            name=args.name,
            verbose=True
        )
        pbar.update(1)

    elapsed = time.time() - start_time
    print(f"\n Train hoan tat sau {elapsed/60:.2f} phut")
    print(f" Ket qua duoc luu tai: {os.path.join(args.project, args.name)}")


if __name__ == "__main__":
    main()
