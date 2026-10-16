import os
import argparse
import subprocess
import glob
from PIL import Image


def run_prediction(model_path, image_path):
    """
    Runs YOLO prediction on a given image and displays the result.
    """

    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model_path}",
        f"source={image_path}",
        "save=True",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running YOLO prediction:", e)
        return

    predict_runs = glob.glob("runs/detect/predict*")
    if not predict_runs:
        print("No prediction runs found.")
        return

    latest_run = max(predict_runs, key=os.path.getctime)

    image_name = os.path.basename(image_path)
    output_image_path = os.path.join(latest_run, image_name)

    if os.path.exists(output_image_path):
        print(f"\nPrediction saved at: {output_image_path}")

        img = Image.open(output_image_path)
        img.show()
    else:
        print(f"Predicted image not found at {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO prediction on an image")

    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to YOLO model weights",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Image path does not exist.")
        return

    if not os.path.exists(args.model):
        print("Model path does not exist.")
        return

    run_prediction(args.model, args.image)


if __name__ == "__main__":
    main()


# python predict_yolo.py --image "C:\Users\YourName\Pictures\banana.jpg" --model "C:\YOLO\models\my_custom_model.pt"

# python predict_yolo.py --image "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\OkomuDataset\dataset\ope\0bb72b75-878e-4bb3-a885-d66ee2962f97.jpg" --model "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\my_model\train\weights\best.pt"

# python predict_yolo.py --image "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\OkomuDataset\dataset\ope\0bb72b75-878e-4bb3-a885-d66ee2962f97.jpg" --model "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\OkomuDataset\samuel_data_model\100 Epoch\my_model\train_results\weights\best.pt"
# python predict_yolo.py --image "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\OkomuDataset\dataset\ope\0bb72b75-878e-4bb3-a885-d66ee2962f97.jpg" --model "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\RoboflowDataset\Model\my_model\train_results\weights\best.pt"


# python predict_yolo.py --image "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\WhatsApp_Image 2026-04-07.jpeg" --model "C:\Users\Admin\Desktop\ongoing_tasks\Okomu App Development Project\dataset\my_model\train\weights\best.pt"
