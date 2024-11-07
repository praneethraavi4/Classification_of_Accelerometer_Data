## Classification of Accelerometer data

### Usage <br>
1.  The repository can be cloned using:
> git clone https://github.com/praneethraavi4/Classification_of_Accelerometer_Data.git


2. Create a virtual environment using:
>python -m venv venv

3. Activate the environment :
>venv\Scripts\activate

4. Install the libraries using:
> pip install -r requirements.txt

5.  The dataset paths are saved in config.yaml file. Update the dataset paths before running the scripts.<br>
```
healthy_data : "path_to_dftrain.h5" 
Anomaly_data : "path_to_dfvalid.h5"
Anomaly_ground_truth: "path_to_dfvalid_groundtruth.csv" 
unseen_data : "path_to_unseen_dataframe_for_prediction" . If unseen data is not available, Anomaly_data can be used. 
train_size : This is the training data size. I have used 80/20 split.

```
6. For model Fitting and evaluation, run :

> python main.py


7.  For prediction of unseen data,run:

>python inference.py
>

8.  The model.pkl is the saved pre-trained model. scaler_data.pkl, scaler_pca.pkl are the
two saved scalers for scaling the features.

Please refer to the file "Classification_task.ipynb" for EDA and plots. The detailed report regarding the task can be 
found in "Classification of Accelerometer data - Report.pdf"