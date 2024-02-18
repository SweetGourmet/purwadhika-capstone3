# Telco Churn Prediction
* Project ini merupakan tugas capstone module 3 dari Purwadhika sebagai penilaian siswa terkait pemahaman mengenai implementasi machine learning.Pada project ini, saya membangun sebuah model yang dapat membantu perusahaan telco untuk memprediksi customer yang akan churn.
* File project berupa file .ipynb
* Dataset didownload dari link google drive yang disediakan oleh mentor.
* Library yang digunakan yaitu:
  * pandas --> Data Manipulation
  * matplotlib.pyplot --> Visualization
  * seaborn --> Visualization
  * plotly.express --> Visualization
  * numpy --> Math operation
  * scipy.stats --> Pengujian statistika(distribution)
  * sklearn --> model,preprocess,metrics
  * imblearn --> for imbalance class
  * category_encoders --> Encoder
  * xgboost.sklearn --> XGBoost model
  * lightgbm --> LightGBM model
  * time --> calculate model runtime
  * dill --> Deployment Model
 
## Data Description
* Dataset ini memuat informasi customer selama menggunakan layanan di sebuah perusahaan telco.Dataset ini memiliki 4930 baris data dengan 11 kolom, yaitu:
  * Dependents: Whether the customer has dependents or not.(Parent,kids,etc)
  * Tenure: Number of months the customer has stayed with the company.
  * OnlineSecurity: Whether the customer has online security or not.
  * OnlineBackup: Whether the customer has online backup or not.
  * InternetService: Whether the client is subscribed to Internet service.
  * DeviceProtection: Whether the client has device protection or not.
  * TechSupport: Whether the client has tech support or not 
  * Contract: Type of contract according to duration.
  * PaperlessBilling: Bills issued in paperless form.
  * MonthlyCharges: Amount of charge for service on monthly bases.
  * Churn: Whether the customer churns or not.(**Target**)
    
## Analytic Approach
* Pertama akan dilakukan analisis pada features dataset untuk menemukan pola yang membedakan antara customer yang churn dan non churn.
* Setelah itu akan dibangun model klasifikasi yang mampu memprediksi customer yang akan churn atau tidak.

## Metric Evaluation(**Recall**)
* Type I Error : False Positive  
Konsekuensi : Cost yang dikeluarkan untuk mempertahankan customer untuk tetap berlangganan seperti diskon atau promo khusus kurang efisien.
* Type II Error : False Negative  
Konsekuensi : Penurunan pendapatan perusahaan.  
  
Berdasarkan kedua konsekuensi, konsekuensi type I error mengakibatkan perusahaan mengeluarkan biaya yang kurang efisien, namun jika dipikirkan baik-baik biaya tersebut tidak seluruhnya menjadi sia-sia, karena dengan biaya yang dikeluarkan seperti diskon khusus untuk pelanggan yang diprediksi akan berhenti tetapi sebenarnya tidak, diskon ini justru bisa membantu menjaga agar customer tidak mencoba/setidaknya mempertimbangkan untuk berlangganan di perusahaan kompetitor lainnya. Sedangkan konsekuensi dari type II error, penurunan pendapatan akibat customer yang berhenti berlangganan tentunya sangat penting dan harus diatasi terutama customer yang memiliki value tinggi, dan juga untuk menutup loss akibat customer yang berhenti karena tidak terprediksi maka perusahaan memerlukan customer baru untuk hanya menutup loss saja, untuk memperoleh customer baru perusahaan juga harus mengeluarkan biaya lagi untuk promosi,iklan,kampanye,dll sehingga cost dari error ini harus sebisa mungkin diminimalisir.Jadi untuk model ini kita ingin memprediksi sebanyak mungkin kelas positif yang benar dari total kelas positif yang ada dan meminimalisir prediksi kelas negatif yang sebenarnya adalah positif, sehingga metric utama yang akan digunakan adalah recall.

## Models
Pada project ini saya melakukan experiment menggunakan beberapa model, yaitu:
* Logistic Regression --> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* Random Forest Classifier --> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
* LightGBM Classifier --> https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
* XGBoost Classifier --> https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
* SVM --> https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

## Preprocess
* Encoder :
  * OneHotEncoding
  * OrdinalEncoding
* Scaler :
  * MinMaxScaler

## Experimenting
Experiment yang dilakukan:
* Without Resampling
* Oversampling
  * Smote
  * RandomOverSampler
* UnderSampling
  * RandomUnderSampling
 
## Hyperparameter Tuning
* Logistic Regression --> C, solver
* Random Forest Classifier --> max_depth, n_estimators, max_features
* LightGBM Classifier --> max_depth, num_leaves, min_data_in_leaf, num_iteration, learning_rate
* XGBoost Classifier --> n_estimators, max_depth, min_child_weight, learning_rate, gamma, colsample_bytree, subsample
* SVM --> C, gamma, kernel

## Conclusion
* Berdasarkan hasil classification report final model, dapat disimpulkan dengan penggunaan model ini perusahaan dapat mengurangi retention cost (diskon,promo) yang dialokasikan untuk kurang lebih 51% dari total customer yang tidak churn, dan model juga dapat memprediksi hingga kurang lebih 93% dari total customer yang churn (semua ini berdasarkan recallnya)
* Model ini memiliki ketepatan prediksi customer yang churn sebesar kurang lebih 41% (precision), jadi dari seluruh prediksi model yang memprediksi customer akan churn hanya terdapat kurang lebih 41%nya yang benar-benar churn. Masih ada customer yang sebenarnya tidak churn tapi diprediksi churn sekitar 49% dari total customer yang tidak churn.(recall)
