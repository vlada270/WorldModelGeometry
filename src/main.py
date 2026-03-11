from src.dataset.nuscenes_loader import NuScenesLoader
from src.dataset.frame_sampler import FrameSampler

DATAROOT = "data/v1.0-mini"


def main():
    loader = NuScenesLoader(dataroot=DATAROOT)
    sampler = FrameSampler(loader.nusc, camera_name="CAM_FRONT", output_dir="outputs/frame_pairs")

    sampler.export_scene(scene_index=0, max_frames=5, copy_images=True)


if __name__ == "__main__":
    main()