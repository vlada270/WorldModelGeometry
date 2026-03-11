from pathlib import Path
import json
import shutil

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class FrameSampler:
    def __init__(self, nusc, camera_name="CAM_FRONT", output_dir="outputs/frame_pairs"):
        self.nusc        = nusc
        self.camera_name = camera_name          # kept for API compatibility
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_scene(self, scene_index=0, max_frames=None, copy_images=True):
        scene      = self.nusc.scene[scene_index]
        scene_name = scene["name"]
        scene_dir  = self.output_dir / scene_name

        # ── Directory layout ──────────────────────────────────────────────
        images_dir      = scene_dir / "images"
        poses_dir       = scene_dir / "poses"
        intrinsics_dir  = scene_dir / "intrinsics"
        lidar_dir       = scene_dir / "lidar"
        annotations_dir = scene_dir / "annotations"

        for d in [images_dir, poses_dir, intrinsics_dir, lidar_dir, annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Per-camera image sub-dirs
        for cam in CAMERAS:
            (images_dir / cam).mkdir(exist_ok=True)

        sample_token = scene["first_sample_token"]
        frame_idx    = 0
        sample_tokens = []

        while sample_token:
            sample     = self.nusc.get("sample", sample_token)
            frame_name = f"frame_{frame_idx:04d}"
            sample_tokens.append(sample_token)

            # ── 1. All 6 cameras ─────────────────────────────────────────
            for cam in CAMERAS:
                cam_token         = sample["data"][cam]
                sample_data       = self.nusc.get("sample_data", cam_token)
                calibrated_sensor = self.nusc.get(
                    "calibrated_sensor", sample_data["calibrated_sensor_token"]
                )
                ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
                image_path = Path(self.nusc.dataroot) / sample_data["filename"]

                # Images
                if copy_images and image_path.exists():
                    shutil.copy2(image_path, images_dir / cam / f"{frame_name}.jpg")

                # Pose  (written once per primary camera to keep your existing format)
                if cam == self.camera_name:
                    pose_data = {
                        "scene_name":        scene_name,
                        "frame_idx":         frame_idx,
                        "camera_name":       cam,
                        "sample_token":      sample_token,
                        "sample_data_token": cam_token,
                        "image_filename":    str(image_path),
                        "timestamp":         sample_data["timestamp"],
                        "ego_translation":   ego_pose["translation"],
                        "ego_rotation":      ego_pose["rotation"],
                        "camera_translation": calibrated_sensor["translation"],
                        "camera_rotation":   calibrated_sensor["rotation"],
                    }
                    with open(poses_dir / f"{frame_name}.json", "w") as f:
                        json.dump(pose_data, f, indent=2)

                    intrinsic_data = {
                        "scene_name":      scene_name,
                        "frame_idx":       frame_idx,
                        "camera_name":     cam,
                        "camera_intrinsic": calibrated_sensor["camera_intrinsic"],
                    }
                    with open(intrinsics_dir / f"{frame_name}.json", "w") as f:
                        json.dump(intrinsic_data, f, indent=2)

            # ── 2. All-camera calibrations (single file per frame) ────────
            all_calibrations = {}
            for cam in CAMERAS:
                cam_token   = sample["data"][cam]
                sd          = self.nusc.get("sample_data", cam_token)
                cs          = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                all_calibrations[cam] = {
                    "intrinsic":   cs["camera_intrinsic"],
                    "translation": cs["translation"],
                    "rotation":    cs["rotation"],
                }
            lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            all_calibrations["LIDAR_TOP"] = {
                "translation": lidar_cs["translation"],
                "rotation":    lidar_cs["rotation"],
            }
            with open(scene_dir / f"calibrations_{frame_name}.json", "w") as f:
                json.dump(all_calibrations, f, indent=2)

            # ── 3. LiDAR point cloud ──────────────────────────────────────
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data  = self.nusc.get("sample_data", lidar_token)
            lidar_path  = Path(self.nusc.dataroot) / lidar_data["filename"]
            if lidar_path.exists():
                pc = LidarPointCloud.from_file(str(lidar_path))
                # Save as (N, 4) float32 binary  [x, y, z, intensity]
                pc.points.T.astype(np.float32).tofile(
                    str(lidar_dir / f"{frame_name}.bin")
                )

            # ── 4. GT 3D annotations ──────────────────────────────────────
            annotations = []
            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                # nusc.box_velocity returns (vx, vy, vz); take XY
                try:
                    vel = self.nusc.box_velocity(ann_token)[:2].tolist()
                except Exception:
                    vel = [0.0, 0.0]
                annotations.append({
                    "token":          ann_token,
                    "instance_token": ann["instance_token"],
                    "category":       ann["category_name"],
                    "translation":    ann["translation"],
                    "size":           ann["size"],
                    "rotation":       ann["rotation"],
                    "velocity":       vel,
                    "num_lidar_pts":  ann["num_lidar_pts"],
                    "visibility":     ann["visibility_token"],
                })
            with open(annotations_dir / f"{frame_name}.json", "w") as f:
                json.dump(annotations, f, indent=2)

            # ── Advance ───────────────────────────────────────────────────
            frame_idx   += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
            sample_token = sample["next"] if sample["next"] else None

        # ── 5. Frame pairs index ──────────────────────────────────────────
        frame_pairs = [
            {
                "pair_idx": i,
                "frame_a":  f"frame_{i:04d}",
                "frame_b":  f"frame_{i+1:04d}",
                "token_a":  sample_tokens[i],
                "token_b":  sample_tokens[i + 1],
            }
            for i in range(len(sample_tokens) - 1)
        ]
        with open(scene_dir / "frame_pairs.json", "w") as f:
            json.dump(frame_pairs, f, indent=2)

        # ── 6. Scene manifest ─────────────────────────────────────────────
        manifest = {
            "scene_name":    scene_name,
            "scene_token":   scene["token"],
            "description":   scene["description"],
            "num_frames":    frame_idx,
            "num_pairs":     len(frame_pairs),
            "cameras":       CAMERAS,
            "sample_tokens": sample_tokens,
        }
        with open(scene_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Exported {frame_idx} frames from {scene_name} to {scene_dir}")
        print(f"  cameras:     {len(CAMERAS)} × {frame_idx} images")
        print(f"  lidar bins:  {frame_idx}")
        print(f"  annotations: {frame_idx} files")
        print(f"  frame pairs: {len(frame_pairs)}")
        return scene_dir