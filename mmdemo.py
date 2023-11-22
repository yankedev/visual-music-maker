import cv2
import tkinter as tk
from PIL import Image, ImageTk
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import time


class PoseEstimation:
    def __init__(self, config, checkpoint, device="cpu", fps=30):
        self.model = init_model(config, checkpoint, device=device)

        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        self.visualizer.set_dataset_meta(self.model.dataset_meta)

        self.cap = cv2.VideoCapture(0)

        #self.delay = int(1000 / fps)
        self.delay = int(1)
        self.window = tk.Tk()
        self.window.title("Pose Estimation")
        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        self.update_image()

    def update_image(self):
        ret, frame = self.cap.read()
        if ret:
            result = self.estimate_pose(frame)

            image = Image.fromarray(result)
            photo = ImageTk.PhotoImage(image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo

        self.window.after(self.delay, self.update_image)

    def estimate_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        batch_results = inference_topdown(self.model, rgb_frame)
        result = merge_data_samples(batch_results)  # Assuming single frame

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time} seconds")
        pred_instances = result.pred_instances
        keypoints = pred_instances.keypoints[0]  # Assuming single person in the frame
        keypoint_scores = pred_instances.keypoint_scores[0]  # Key point scores

        print("KEYPOINTS: " + str(keypoints))
        #print("SCORE: " + str(keypoint_scores))
        # Skeleton connections for COCO keypoints
        skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head to shoulders
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        for idx, (x, y) in enumerate(keypoints[:, :2]):
            conf = keypoint_scores[idx]
            if conf > 0.3:  # Confidence threshold
                cv2.circle(rgb_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for start, end in skeleton:
            if keypoint_scores[start] > 0.3 and keypoint_scores[end] > 0.3:
                cv2.line(
                    rgb_frame,
                    (int(keypoints[start][0]), int(keypoints[start][1])),
                    (int(keypoints[end][0]), int(keypoints[end][1])),
                    (255, 0, 0),
                    2,
                )
        # visualized_image_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return rgb_frame

    def run(self):
        self.window.mainloop()

    def __del__(self):
        self.cap.release()


if __name__ == "__main__":
    config = "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    checkpoint = "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"

    app = PoseEstimation(config, checkpoint, fps=30)
    app.run()