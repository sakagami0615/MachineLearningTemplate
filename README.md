## **1. 概要**

機械学習を実施する際、データセットを整理したり、学習/テスト/評価スクリプトを書いたりするのが面倒なので、フォーマット化してすぐに学習を実施できる環境を構築する（したい）

※今回はSVM用に作成したが、ほかの学習方式でも対応可能<br>
※画像のデータセットにのみ対応
<br><br>

## **2. フォルダ構成**

- **train**：学習データ用フォルダ
- **test**：テストデータ用フォルダ
- **metrics**：テストデータ評価用フォルダ
- **utility**：上記の処理で必要になるスクリプト用フォルダ

※どのデータセットに対して処理を実施するかは、「param.json」に記載する<br>
<br>

## **3. 「utility/dataset.py」に関して**

データセットの構成の差異を吸収するための作成したスクリプト<br>
データセットごとに「dataset_info.json」を作成する必要がある
<br><br>

## **2.1　「dataset_info.json」の記載例**

下記のようなデータセットの場合は、下記に示すようになる<br>
- 犬と猫の画像を持つ「SampleDataset」データセット<br>
- 学習画像は、フォルダでラベルを管理<br>
- テスト画像は、ファイル名とラベルの対応表「correct_label.csv」を持つ<br><br>

```python
{
	"name": "SampleDataset",
	"train":{
		"folder": "train",
		"labels": {
			"dog": 1,
			"cat": 2
		},
		"convert":{
			"type": "label",
			"ext": ["jpg", "png"]
		}
	},
	"test":{
		"folder": "test/img",
		"labels": {
			"dog": 1,
			"cat": 2
		},
		"convert":{
			"type": "correct",
			"file": "test/correct_label.csv"
		}
	}
}
```

**"name"**：データセットの名前<br>
　どのデータセットのinfoかを把握するものであり、処理では使用しません<br>

**"train"**：学習用のデータ関連のパラメータ<br>
　上記で説明したパターンに応じてパラメータを指定する<br>

**"test"**：ラベルの振り方を指定するパラメータ関連<br>
　上記で説明したパターンに応じてパラメータを指定する<br>
<br>

"train"、"test"内のパラメータの記載によってフォルダ構成の差異を吸収する<br>
詳細は下記の節で示す
<br><br>

## **2.2. フォルダ構成の際によるパラメータ指定方法**

- **[Pattern 1]フォルダでラベルを振っているパターン**

	**"folder"**：フォルダ名を指定<br>

	**"labels"**：ラベル名とラベルIDの対応辞書<br>

	**"convert"**：ラベルの振り方を指定するパラメータ関連<br>
	
	- **"type"**：今回の場合では、"label"とする<br>
	- **"ext"**：読み込み対象のファイル拡張子リスト<br>

	「folder」で指定したフォルダ内に「labels」で用意したラベル名のフォルダに学習データが保管されている必要がある<br>
	※フォルダツリーはこんな感じ
	```
	FolderName
	├─labelName1
	│ 	    image_01.jpg
	│  	    image_01.jpg
	│
	└─labelName2
	        image_01.jpg
  	        image_01.jpg
	```
	
	下記、パラメータのサンプル
	```python
	{
		"folder": "FolderName",
		"labels": {
			"labelName1": 1,
			"labelName2": 2
		},
		"convert":{
			"type": "label",
			"ext": ["jpg", "png"]
		}
	}
	```
	<br>

- **[Pattern 2]ファイル名でラベルを振っているパターン**

	**"folder"**：フォルダ名を指定<br>

	**"labels"**：ラベル名とラベルIDの対応辞書<br>

	**"convert"**：ラベルの振り方を指定するパラメータ関連<br>
	
	- **"type"**：今回の場合では、"file"とする<br>
	- **"ext"**：読み込み対象のファイル拡張子リスト<br>
	- **"pattern"**：ファイル名に振っているラベルのフォーマット指定<br>
		{ラベル名}.{番号}.{拡張子} の順で指定する方法になっている<br>
		※ファイル名が「labelName1_01.jpg」のような場合は、<br>
		　フォーマットは、"{}_{}.{}" のようになる

	※フォルダツリーはこんな感じ
	```
	FolderName
	 	    labelName1.01.jpg
	  	    labelName1.02.jpg
			labelName2.01.jpg
	  	    labelName2.02.jpg
	```
	
	下記、パラメータのサンプル
	```python
	{
		"folder": "FolderName",
		"labels": {
			"labelName1": 1,
			"labelName2": 2
		},
		"convert":{
			"type": "file",
			"pattern": "{}.{}.{}",
			"ext": ["jpg", "png"]
		}
	}
	```
	<br>

- **[Pattern 3]ファイル名とラベルの対応表があるパターン**

	**"folder"**：フォルダ名を指定<br>

	**"labels"**：ラベル名とラベルIDの対応辞書<br>

	**"convert"**：ラベルの振り方を指定するパラメータ関連<br>
	
	- **"type"**：今回の場合では、"correct"とする<br>
	- **"file"**：ファイル名とラベルの対応表ファイル<br>
		下記の表を記載しているCSVファイルを指定する<br>
		```
		name	label
		1.jpg	1
		2.jpg	1
		3.jpg	2
		4.jpg	2
		```

	※フォルダツリーはこんな感じ
	```
	FolderName
	 	    1.jpg
	  	    2.jpg
			3.jpg
	  	    4.jpg
			correct_label.csv
	```

	下記、パラメータのサンプル
	```python
	{
		"folder": "FolderName",
		"labels": {
			"labelName1": 1,
			"labelName2": 2
		},
		"convert":{
			"type": "correct",
			"file": "FolderName/correct_label.csv"
		}
	}
	```
	<br>

