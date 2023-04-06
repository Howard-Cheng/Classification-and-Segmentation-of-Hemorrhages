{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d9e25cb-a090-445f-9f15-7f2f339e7c26",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow import keras\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score,confusion_matrix,classification_report\n",
    "import numpy as np\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b4271df-014b-45ba-a06f-05c4f2468c3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.chdir('/Users/howard/Desktop/Renders/')\n",
    "os.getcwd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44869ef7-b4b3-407a-8968-b5790de6b47f",
   "metadata": {},
   "outputs": [],
   "source": [
    "file_dir = '/Users/howard/Desktop/Renders/'\n",
    "labels = pd.read_csv(file_dir + 'hemorrhage-labels.csv')\n",
    "display(labels)\n",
    "y = labels.epidural\n",
    "y = (y*2).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a3b2ffd-7937-44b6-974b-29d7860e3412",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = np.zeros([282306, 262144])\n",
    "\n",
    "for n, file_name in enumerate(labels.Image):\n",
    "    data[n,:] = np.mean(matplotlib.image.imread(file_dir + file_name+'.jpg'),axis=2).reshape(-1)\n",
    "# Now we can use the file name to read the data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "807a2a92-2592-4bc0-882e-1f8d265fc4ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train, y_train), (x_test, y_test) = data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00d24303-185b-4f4a-8b89-1f00b266752f",
   "metadata": {},
   "source": [
    "### Data Pre-processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8a74ea7d-49d1-4dd7-ab40-100eac708581",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalization\n",
    "x_train = x_train/255.0\n",
    "x_test = x_test/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2697dd06-2eb8-4262-bf1f-5b8269453d7f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 3072)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#so,eventually,model.predict() should also be a 2d input\n",
    "nsamples, nx, ny, nrgb = x_test.shape\n",
    "x_test2 = x_test.reshape((nsamples,nx*ny*nrgb))\n",
    "x_test2.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22825802-4001-4ee0-8fcf-23865f879245",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Implementing a Decision Tree Classfier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "5b1620f2-0778-4b24-b6b0-1f337175986f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a923bd41-58bd-4ffe-9ee7-5450ad5c5aea",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtc=DecisionTreeClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fec8cbba-153a-4c5c-89d2-30950a9af720",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',\n",
       "                       max_depth=None, max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, presort='deprecated',\n",
       "                       random_state=None, splitter='best')"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dtc.fit(x_train2,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b5f63b33-d32a-4175-85ad-9f201906723c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7, 1, 9, ..., 2, 5, 1], dtype=uint8)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred_dtc=dtc.predict(x_test2)\n",
    "y_pred_dtc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "0c160e22-be9f-40bc-9fa9-91fc13085af1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.34      0.33      0.33      1028\n",
      "           1       0.27      0.28      0.27       936\n",
      "           2       0.22      0.21      0.21      1040\n",
      "           3       0.19      0.18      0.18      1013\n",
      "           4       0.23      0.22      0.23      1062\n",
      "           5       0.22      0.22      0.22       986\n",
      "           6       0.29      0.30      0.29       968\n",
      "           7       0.27      0.29      0.28       937\n",
      "           8       0.38      0.37      0.38      1034\n",
      "           9       0.29      0.29      0.29       996\n",
      "\n",
      "    accuracy                           0.27     10000\n",
      "   macro avg       0.27      0.27      0.27     10000\n",
      "weighted avg       0.27      0.27      0.27     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "accuracy_score(y_pred_dtc,y_test)\n",
    "print(classification_report(y_pred_dtc,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "c4e431d0-e076-4c96-b740-abd9e3539a86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[338,  79, 100,  60,  59,  44,  42,  80, 139,  87],\n",
       "       [ 75, 266,  40,  52,  44,  65,  45,  69, 106, 174],\n",
       "       [ 90,  56, 219, 105, 165, 102, 121,  84,  44,  54],\n",
       "       [ 59,  74, 105, 186,  95, 151, 123, 105,  53,  62],\n",
       "       [ 68,  69, 142, 118, 235,  96, 135,  99,  53,  47],\n",
       "       [ 50,  59, 113, 140,  97, 218,  97, 104,  45,  63],\n",
       "       [ 35,  54, 105, 121, 119, 104, 290,  56,  30,  54],\n",
       "       [ 46,  62,  84,  96, 106,  89,  62, 269,  42,  81],\n",
       "       [158, 109,  42,  62,  30,  62,  32,  63, 384,  92],\n",
       "       [ 81, 172,  50,  60,  50,  69,  53,  71, 104, 286]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_pred_dtc,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b57a09ca-45ef-44d4-a030-74e99713b26e",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Random Forest Classfier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "58c1cf82-3d29-452f-a9b3-bb956685c7e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "895c6695-65a8-4e36-a183-a27b73ef807b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model2=RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "107812e7-3a27-4bf9-9610-60e1d54354fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='gini', max_depth=None, max_features='auto',\n",
       "                       max_leaf_nodes=None, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=100,\n",
       "                       n_jobs=None, oob_score=False, random_state=None,\n",
       "                       verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.fit(x_train2,y_train.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "6ad5385c-ff64-4872-9580-acf5c858b302",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 8, 8, ..., 5, 3, 7], dtype=uint8)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model2.predict(x_test2)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e1d70225-b0e0-4491-93f9-2ace99d49154",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.55      0.54      0.54      1034\n",
      "           1       0.56      0.52      0.54      1076\n",
      "           2       0.33      0.36      0.34       896\n",
      "           3       0.26      0.33      0.29       780\n",
      "           4       0.38      0.38      0.38       989\n",
      "           5       0.40      0.41      0.40       974\n",
      "           6       0.57      0.47      0.51      1209\n",
      "           7       0.45      0.52      0.48       862\n",
      "           8       0.59      0.57      0.58      1033\n",
      "           9       0.54      0.47      0.50      1147\n",
      "\n",
      "    accuracy                           0.46     10000\n",
      "   macro avg       0.46      0.46      0.46     10000\n",
      "weighted avg       0.47      0.46      0.47     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "accuracy_score(y_pred,y_test)\n",
    "print(classification_report(y_pred,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "81799547-a123-4b68-a530-cc1f4e02257b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[554,  34, 101,  56,  59,  32,  14,  44,  93,  47],\n",
       "       [ 40, 557,  44,  37,  19,  26,  32,  44,  95, 182],\n",
       "       [ 67,  29, 326,  65, 165,  85,  69,  55,  20,  15],\n",
       "       [ 21,  32,  75, 260,  54, 154,  66,  50,  35,  33],\n",
       "       [ 21,  15, 149,  88, 378,  81, 120, 107,  12,  18],\n",
       "       [ 24,  26,  72, 193,  55, 395,  61,  86,  36,  26],\n",
       "       [ 19,  36, 115, 149, 148,  89, 567,  49,  13,  24],\n",
       "       [ 26,  25,  62,  65,  80,  78,  24, 446,  24,  32],\n",
       "       [164,  63,  26,  18,  23,  27,   9,  27, 593,  83],\n",
       "       [ 64, 183,  30,  69,  19,  33,  38,  92,  79, 540]])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_pred,y_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ddf1a60d-f27d-4212-ae3c-1397a7ad2e7c",
   "metadata": {},
   "source": [
    "## Adaboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7016e7e-2e5a-423f-a85b-3e063b0ec144",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "import numpy as np\n",
    "\n",
    "rng = np.random.RandomState(1)\n",
    "\n",
    "boosted = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=rng)\n",
    "boosted.fit(x_train2,y_train.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "cd67de3f-01ed-4aca-bdc1-12935e43f787",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.3308"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# boosted performance\n",
    "boosted.score(x_test2, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "f929ea2f-14a8-42f4-a564-315e7c2fb75c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score, roc_curve, roc_auc_score\n",
    "import seaborn as sns "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "fe7f5069-bec0-41bc-ba82-6b1552df8d96",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "multiclass format is not supported",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-51-89c6333cad3a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0my_pred_proba\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mboosted\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict_proba\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_test2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mfpr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtpr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mt\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mroc_curve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred_proba\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mauc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mroc_auc_score\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred_proba\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/sklearn/metrics/_ranking.py\u001b[0m in \u001b[0;36mroc_curve\u001b[0;34m(y_true, y_score, pos_label, sample_weight, drop_intermediate)\u001b[0m\n\u001b[1;32m    768\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    769\u001b[0m     \"\"\"\n\u001b[0;32m--> 770\u001b[0;31m     fps, tps, thresholds = _binary_clf_curve(\n\u001b[0m\u001b[1;32m    771\u001b[0m         y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)\n\u001b[1;32m    772\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/sklearn/metrics/_ranking.py\u001b[0m in \u001b[0;36m_binary_clf_curve\u001b[0;34m(y_true, y_score, pos_label, sample_weight)\u001b[0m\n\u001b[1;32m    534\u001b[0m     if not (y_type == \"binary\" or\n\u001b[1;32m    535\u001b[0m             (y_type == \"multiclass\" and pos_label is not None)):\n\u001b[0;32m--> 536\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"{0} format is not supported\"\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_type\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    537\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    538\u001b[0m     \u001b[0mcheck_consistent_length\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_score\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: multiclass format is not supported"
     ]
    }
   ],
   "source": [
    "y_pred_proba = boosted.predict_proba(x_test2)[:,1]\n",
    "\n",
    "fpr, tpr, t = roc_curve(y_test, y_pred_proba)\n",
    "auc = roc_auc_score(y_test, y_pred_proba)\n",
    "\n",
    "plt.plot(fpr, tpr, label=\"AUC\"+str(auc))\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "y_pred_RF = boosted.predict(x_test2)\n",
    "\n",
    "print(classification_report(y_test, y_pred_RF))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fb6bca2-bc57-4ffc-9b34-9ecc7c3541e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred_RF)\n",
    "# print(cnf_matrix)\n",
    "\n",
    "cf_df = pd.DataFrame(cnf_matrix, columns=['0', '1'], index=['0', '1'])\n",
    "sns.set(font_scale=1.4)\n",
    "plt.figure(figsize=(7,5))\n",
    "sns.heatmap(cf_df, annot=True, fmt=\"5.0f\")\n",
    "plt.title('Confusion Matrix')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f877b900-dce0-4929-a103-30e05232320a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "params = {\n",
    " 'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450],\n",
    " 'learning_rate' : [0.01, 0.05, 0.1, 0.5]\n",
    " }\n",
    "\n",
    "search = RandomizedSearchCV(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=rng), params, cv=5, return_train_score=True)\n",
    "search.fit(x_train2,y_train.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af7751f4-ac40-4cfd-9544-7e3cd8de3546",
   "metadata": {},
   "outputs": [],
   "source": [
    "params = search.best_params_\n",
    "tuned_boosted = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=rng, n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])\n",
    "tuned_boosted.fit(x_train2,y_train.ravel())\n",
    "tuned_boosted.score(x_test2, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd7ad29a-e2de-435d-9baf-a3eec0093b01",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
