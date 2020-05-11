import os
import re
import glob
import json
import pandas
import itertools


class Dataset:

	def __init__(self, parent_path, dataset_name):
		
		self.dataset_path = os.path.join(parent_path, dataset_name)

		with open(os.path.join(self.dataset_path, 'dataset_info.json')) as f:
			self.dataset_info_json = json.load(f)


	def __GetPathLabelList_ModeLabel(self, info_dict):
		
		label_dict = info_dict['labels']
		folder_name = info_dict['folder']
		search_pattern = r'/*\.({})'.format('|'.join(info_dict['convert']['ext']))
		folder_path_list = [os.path.join(self.dataset_path, folder_name, label) for label in label_dict.keys()]

		img_path_list_nd = [
			[img_path for img_path in glob.glob('{}/*'.format(folder_path)) if re.search(search_pattern, str(img_path))]
			for folder_path in folder_path_list]
		img_label_list_nd = [[label_dict[label]]*len(img_paths) for (img_paths, label) in zip(img_path_list_nd, label_dict.keys())]
		
		img_path_list = list(itertools.chain.from_iterable(img_path_list_nd))
		img_label_list = list(itertools.chain.from_iterable(img_label_list_nd))
		is_load = True
		return (is_load, img_path_list, img_label_list)


	def __GetPathLabelList_ModeFile(self, info_dict):
		
		label_dict = info_dict['labels']
		folder_name = info_dict['folder']
		folder_path = os.path.join(self.dataset_path, folder_name)
		search_pattern = r'/*({})*\.({})'.format('|'.join(label_dict.keys()), '|'.join(info_dict['convert']['ext']))
		
		def GetLabel(img_path, label_dict):
			img_name = os.path.basename(img_path)
			for label in label_dict.keys():
				if label in img_name:
					return label_dict[label]
			return -1

		img_path_list = [img_path for img_path in glob.glob('{}/*'.format(folder_path)) if re.search(search_pattern, str(img_path))]
		img_label_list = [GetLabel(img_path, label_dict) for img_path in img_path_list]
		is_load = True
		return (is_load, img_path_list, img_label_list)


	def __GetPathLabelList_ModeCorrect(self, info_dict):

		folder_name = info_dict['folder']
		file_path = os.path.join(self.dataset_path, info_dict['convert']['file'])
		convert_dataframe = pandas.read_csv(file_path)

		img_path_list = [os.path.join(self.dataset_path, folder_name, name) for name in convert_dataframe['name']]
		img_label_list = [label for label in convert_dataframe['label']]
		is_load = True
		return (is_load, img_path_list, img_label_list)


	def LoadPathLabelList(self, load_key):
		
		info_dict = self.dataset_info_json[load_key]

		if info_dict['convert']['type'] == 'label':
			is_load, img_path_list, img_label_list = self.__GetPathLabelList_ModeLabel(info_dict)

		elif info_dict['convert']['type'] == 'file':
			is_load, img_path_list, img_label_list = self.__GetPathLabelList_ModeFile(info_dict)

		elif info_dict['convert']['type'] == 'correct':
			is_load, img_path_list, img_label_list = self.__GetPathLabelList_ModeCorrect(info_dict)

		else:
			img_path_list = None
			img_label_list = None
			is_load = False

		return (is_load, img_path_list, img_label_list)
	

	def GetLabelDict(self, load_key):

		return self.dataset_info_json[load_key]['labels']
