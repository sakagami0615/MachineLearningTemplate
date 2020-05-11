
import sys
import json
import numpy
import tqdm
import pandas
import pickle
from sklearn import svm

sys.path.append('../')

from utility.dataset import Dataset
from utility.interface import *


PARAM_FILE_PATH = '../param.json'
TRAIN_CLF_FILEPATH = '../train/train_clf.pkl'


if __name__ == '__main__':

	with open(PARAM_FILE_PATH) as f:
		param_json = json.load(f)
	dataset = Dataset(param_json['test']['parent_path'], param_json['test']['dataset_name'])

	print('■ Get Path and Label')
	is_load, img_path_list, img_label_list = dataset.LoadPathLabelList('test')
	if not is_load: sys.exit(-1)
	
	print('■ feature extraction')
	img_desc_list = [ExtractFeature(ReadShapeImage(img_path)) for img_path in tqdm.tqdm(img_path_list)]

	test_datas = numpy.array(img_desc_list)
	test_labels = numpy.array(img_label_list)

	print('■ test')
	clf = pickle.load(open(TRAIN_CLF_FILEPATH, 'rb'))
	predicts = clf.predict(test_datas)

	print('■ save predict')
	dataframe = pandas.DataFrame({
		'name' : img_path_list,
		'label' : test_labels,
		'predict' : predicts})
	dataframe.to_csv('predict.csv')