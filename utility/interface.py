from skimage import io
from skimage.transform import resize
from skimage.feature import hog


def ReadShapeImage(img_path):

	img = io.imread(img_path, as_gray=True)
	img = resize(img, (100, 100))
	return img


def ExtractFeature(img):

	# HOG
	desc = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm='L2')
	return desc