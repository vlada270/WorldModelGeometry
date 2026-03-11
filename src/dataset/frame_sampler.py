from pathlib import Path
import json
import shutil


class FrameSampler:
    def __init__(self, nusc, camera_name="CAM_FRONT", output_dir="outputs/frame_pairs"):
        self.nusc = nusc
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_scene(self, scene_index=0, max_frames=None, copy_images=True):
        scene = self.nusc.scene[scene_index]
        scene_name = scene["name"]
        scene_dir = self.output_dir / scene_name
        images_dir = scene_dir / "images"
        poses_dir = scene_dir / "poses"
        intrinsics_dir = scene_dir / "intrinsics"

        images_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_dir.mkdir(parents=True, exist_ok=True)

        sample_token = scene["first_sample_token"]
        frame_idx = 0

        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            cam_token = sample["data"][self.camera_name]
            sample_data = self.nusc.get("sample_data", cam_token)

            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

            image_path = Path(self.nusc.dataroot) / sample_data["filename"]

            frame_name = f"frame_{frame_idx:04d}"
            target_image_path = images_dir / f"{frame_name}.jpg"

            if copy_images:
                shutil.copy2(image_path, target_image_path)

            pose_data = {
                "scene_name": scene_name,
                "frame_idx": frame_idx,
                "camera_name": self.camera_name,
                "sample_token": sample_token,
                "sample_data_token": cam_token,
                "image_filename": str(image_path),
                "timestamp": sample_data["timestamp"],
                "ego_translation": ego_pose["translation"],
                "ego_rotation": ego_pose["rotation"],
                "camera_translation": calibrated_sensor["translation"],
                "camera_rotation": calibrated_sensor["rotation"],
            }

            intrinsic_data = {
                "scene_name": scene_name,
                "frame_idx": frame_idx,
                "camera_name": self.camera_name,
                "camera_intrinsic": calibrated_sensor["camera_intrinsic"],
            }

            with open(poses_dir / f"{frame_name}.json", "w") as f:
                json.dump(pose_data, f, indent=2)

            with open(intrinsics_dir / f"{frame_name}.json", "w") as f:
                json.dump(intrinsic_data, f, indent=2)

            frame_idx += 1

            if max_frames is not None and frame_idx >= max_frames:
                break

            sample_token = sample["next"]

        print(f"Exported {frame_idx} frames from {scene_name} to {scene_dir}")
        return scene_dir