import sys
import json
import numpy
import tqdm
import pickle
from sklearn import svm

sys.path.append('../')

from utility.dataset import Dataset
from utility.interface import *


PARAM_FILE_PATH = '../param.json'


if __name__ == '__main__':

	with open(PARAM_FILE_PATH) as f:
		param_json = json.load(f)
	dataset = Dataset(param_json['train']['parent_path'], param_json['train']['dataset_name'])

	print('■ Get Path and Label')
	is_load, img_path_list, img_label_list = dataset.LoadPathLabelList('train')
	if not is_load: sys.exit(-1)

	print('■ feature extraction')
	img_desc_list = [ExtractFeature(ReadShapeImage(img_path)) for img_path in tqdm.tqdm(img_path_list)]

	train_data = numpy.array(img_desc_list)
	train_label = numpy.array(img_label_list)

	print('■ train')
	clf = svm.SVC(gamma=0.001)
	clf.fit(train_data, train_label)

	pickle.dump(clf, open('train_clf.pkl', 'wb'))
