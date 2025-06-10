import sys
import struct
import collections
import numpy as np
from argparse import ArgumentParser, Namespace
from exp_runner import Runner

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)



def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def create_camera_to_world_matrix(qvec, tvec):
    """
    Creates a 4x4 camera-to-world matrix from a quaternion and a translation vector.

    Args:
        qvec (np.ndarray or list): Quaternion (w, x, y, z) representing the rotation
                                   from camera coordinates to world coordinates.
                                   COLMAP's convention is (QW, QX, QY, QZ).
        tvec (np.ndarray or list): Translation vector (tx, ty, tz) representing the
                                   position of the camera origin in world coordinates.

    Returns:
        np.ndarray: A 4x4 camera-to-world transformation matrix.
    """
    if not isinstance(qvec, np.ndarray):
        qvec = np.array(qvec)
    if not isinstance(tvec, np.ndarray):
        tvec = np.array(tvec)

    if qvec.shape != (4,):
        raise ValueError("Quaternion (qvec) must have 4 elements (w, x, y, z).")
    if tvec.shape != (3,):
        raise ValueError("Translation vector (tvec) must have 3 elements (tx, ty, tz).")

    # Convert quaternion to 3x3 rotation matrix
    # This rotation matrix rotates points from camera frame to world frame
    R_cam_to_world = qvec2rotmat(qvec)

    # Create the 4x4 transformation matrix
    # This matrix transforms points from camera coordinates to world coordinates
    cam_to_world_matrix = np.eye(4)
    cam_to_world_matrix[:3, :3] = R_cam_to_world
    cam_to_world_matrix[:3, 3] = tvec

    return cam_to_world_matrix


def training(args):
  runner = Runner("./confs/neuris.conf", "scene0625_00", "train", "", False, -1, args)
  print(runner.base_exp_dir)
  #runner.writer = SummaryWriter(log_dir=)
  runner.update_learning_rate()

if __name__ == "__main__":
  parser = ArgumentParser(description="Run NeuRIS training")
  parser.add_argument("--scene_name", type=str, default="scene0625_00")
  parser.add_argument("--exp_name", type=str, default="exp_scene0625_00")
  parser.add_argument("--exp_dir", type=str, default="./test")
  parser.add_argument("--data_dir", type=str, default="C:/Users/JTStephens/Downloads/replica/replica/test/office_0_1")
  parser.add_argument("--iterations", type=int, default=20000)
  parser.add_argument("--is_sdf_norm", type=bool, default=True)

  args = parser.parse_args(sys.argv[1:])
  # cameras = read_cameras_binary("C:/Users/JTStephens/Downloads/replica/replica/test/office_0_1/sparse/0/cameras.bin")
  # print("Cameras loaded:", cameras)
  # images = read_images_binary("C:/Users/JTStephens/Downloads/replica/replica/test/office_0_1/sparse/0/images.bin")
  # world_mat = []
  # for img in images.items():
  #     print(img)
  #     print("\n")
  #     world_mat.append(create_camera_to_world_matrix(img[1].qvec, img[1].tvec))

  # print(f"{len(world_mat)} world matrices created.")
  # print(world_mat[0])

  # scale_mat =  np.array([np.eye(4) for _ in range(len(world_mat))])
  # print(f"{len(scale_mat)} scale matrices created.")
  # print(scale_mat[0])

  # for cam_id, cam in cameras.items():
  #   print(f"Camera ID: {cam_id}, Model: {cam.model}, Width: {cam.width}, Height: {cam.height}, Params: {cam.params}")
  training(args)