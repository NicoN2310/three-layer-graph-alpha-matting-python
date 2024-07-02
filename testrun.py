import numpy as np
from PIL import Image
from three_layer_graph_alpha_matting.matting import calculate_matte


# Set paths to image, trimap and alpha matte
path_to_image = "test_image.png"
path_to_trimap = "test_trimap.png"
path_to_matte = "test_matte.png"
path_to_cutout = "test_cutout.png"

# Load image and trimap
image = np.array(Image.open(path_to_image).convert("RGB"), dtype=np.float64)
trimap = np.array(Image.open(path_to_trimap).convert("L"), dtype=np.float64)

# Calculate alpha matte
matte = calculate_matte(image, trimap)

# Save alpha matte
Image.fromarray((matte * 255).astype(np.uint8), mode="L").save(path_to_matte)

# Save cutout
cutout = (image * matte[:, :, None]).astype(np.uint8)
Image.fromarray(cutout).save(path_to_cutout)
