from sklearn import metrics
import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def Calc(labels, predicts, label_list):
	
	con_matrix_data = metrics.confusion_matrix(labels, predicts)
	con_matrix = pandas.DataFrame(con_matrix_data, index=label_list, columns=label_list)
	
	return con_matrix


def Save(save_path, con_matrix):
	
	sns.heatmap(con_matrix)
	plt.savefig(save_path)