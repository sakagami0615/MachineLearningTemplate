import sys
import json
import numpy
import pandas

sys.path.append('../')

from utility.dataset import Dataset
import cm
import roc



PARAM_FILE_PATH = '../param.json'
PREDICT_FILEPATH = '../test/predict.csv'


if __name__ == '__main__':

	with open(PARAM_FILE_PATH) as f:
		param_json = json.load(f)
	dataset = Dataset(param_json['metrics']['parent_path'], param_json['metrics']['dataset_name'])

	convert_dataframe = pandas.read_csv(PREDICT_FILEPATH)

	label_list = [label for label in convert_dataframe['label']]
	predict_list = [predict for predict in convert_dataframe['predict']]

	# ラベルまだ振れてない箇所を除外する処理
	predict_list = [predict for (predict, label) in zip(predict_list, label_list) if label != -1]
	label_list = [label for label in label_list if label != -1]

	label_dict_keys = list(dataset.GetLabelDict('train').keys())
	labels = numpy.array(label_list)
	predicts = numpy.array(predict_list)
	
	cm_matrix = cm.Calc(labels, predicts, label_dict_keys)
	cm.Save('cm.png', cm_matrix)

	fpr, tpr, thresholds, auc = roc.Calc(labels, predicts, 2)
	roc.Save('roc.png', fpr, tpr, auc)
	