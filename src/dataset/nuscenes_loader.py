from nuscenes.nuscenes import NuScenes


class NuScenesLoader:

    def __init__(self, dataroot, version="v1.0-mini", verbose=True):
        self.nusc = NuScenes(
            version=version,
            dataroot=dataroot,
            verbose=verbose
        )

    def get_scene_names(self):
        return [scene["name"] for scene in self.nusc.scene]

    def get_scene(self, index=0):
        return self.nusc.scene[index]