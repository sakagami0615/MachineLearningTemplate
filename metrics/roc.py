from sklearn import metrics
import matplotlib.pyplot as plt


def Calc(labels, predicts, pos_label):

	# FPR, TPR, AUC の算出
	fpr, tpr, thresholds = metrics.roc_curve(labels, predicts, pos_label=pos_label)
	auc = metrics.auc(fpr, tpr)

	return fpr, tpr, thresholds, auc


def Save(save_path, fpr, tpr, auc):

	# ROC曲線プロット
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
	plt.legend()
	plt.title('ROC curve')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.grid(True)
	plt.savefig(save_path)