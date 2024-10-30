# from predict import Predictor
# from PIL import Image
# import numpy as np
# import io

# obj = Predictor()

# img_path = "standing_man.jpg" # 이미지 경로
# image = Image.open(img_path)

# ### upload a source file
# buffered = io.BytesIO()
# image.save(buffered, format="PNG")
# img_value = buffered.getvalue()
# upload_binary_img = {
#   img_path: img_value
# }
# Path = list(upload_binary_img.keys())[0] # Please check "/content/PIDM/data/" for sample images for source file.
# # print(Path)
# ### OR enter the path
# #Path = "data/deepfashion_256x256/target_edits/reference_img_5.png"

# obj.predict_pose(image=Path, sample_algorithm='ddim', num_poses=4, nsteps=50)

# Image('output.png')

from predict import Predictor

obj = Predictor()
path_img = "./standing_man.jpg"

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

original = numpy.load('/Users/dgsw8th46/Downloads/PIDM/data/deepfashion_256x256/target_pose/joint_data.npy')
original = numpy.load('/Users/dgsw8th46/Downloads/PIDM/data/deepfashion_256x256/target_pose/joint_data.npy')

# print(original.shape)
# print(new.shape)

# print(original.dtype)
# print(new.dtype)

# print(original)
obj.predict_pose(image=path_img, sample_algorithm='ddim', num_poses=4, nsteps=50)