# HappyML Introduction
HappyML is a machine learning library for educational purpose.  This library simplified many aspects of machine learning including preprocessing, model creation, data visualization...etc.  This library is more experimental and not recommended for production purpose.

HappyML 是一個教學用的機器學習函式庫。該函式庫簡化了機器學習的程式碼寫作。包括：前處理、模型建置、資料視覺化...等方面。本函式庫乃教學用途，實驗性質重，可能存在不少潛在錯誤。不建議直接使用在正式場合。



# Revision History

* 2021/06/03-01
  * [FIX] cluster_drawer() shows error message: "findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans"
* 2021/05/26-01
  * [FIX] PCASelector.fit() shows same information no matter verbose=True or =False.
* 2021/05/19-01
  * [REMOVE] Remove auto=False parameter from KBestSelector.fit()
  * [FIX] KBestSelector.fit() show nothing when <>auto and verbose=True.
  * [FIX] Remove iid= parameter from GridSearchCV()
* 2020/08/03-01
  * [NEW] Add show_first_n_images() into model_drawer.py
  * [NEW] Add create_seq_model() into neural_networks.py
* 2020/07/26-01
  * [NEW] Add epochs_metrics_plot() into model_drawer.py
* 2020/07/26-01
  * [NEW] Create this repository.
