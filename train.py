from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def FFT(xreal, ximag):    
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2
    
    p = int(math.log(n, 2))
    
    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
            
    wreal = []
    wimag = []
        
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    
    wreal.append(float(1.0))
    wimag.append(float(0.0))
    
    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
        
    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2
        
    return n, xreal, ximag   
    
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(a) / len(a))
    
    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2]
    g_kurtosis = [g_s1 / g_s2]
    a_skewness = [a_k1 / a_k2]
    g_skewness = [g_k1 / g_k2]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
        
    
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    writer.writerow(output)

def data_generate(datapath, tar_dir):
    # datapath = './train_data'
    # tar_dir = 'tabular_data_train'
    pathlist_txt = Path(datapath).glob('**/*.txt')

    
    for file in pathlist_txt:
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype = int)
        # filename.append(int(Path(file).stem))
        # all_swing.append([swing_index])

        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                
        

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except:
                print(Path(file).stem)
                continue

def predict_new_data(models, label_encoders, scaler, group_size=27):
    data_generate('./test_data', 'tabular_data_test')
    # 讀取新的 info
    test_info = pd.read_csv('./test_data')
    datapath = './tabular_data_test'
    datalist = list(Path(datapath).glob('**/*.csv'))

    x_test = pd.DataFrame()
    y_info = []  # 存 unique_id, player_id 做 mapping
    
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = test_info[test_info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        x_test = pd.concat([x_test, data], ignore_index=True)
        y_info.extend([{'unique_id': unique_id, 'player_id': player_id}] * len(data))

    X_test_scaled = scaler.transform(x_test)
    df_y_info = pd.DataFrame(y_info)

    # 初始化結果表
    final_results = {}

    for col in ['gender', 'hold racket handed', 'play years', 'level']:
        clf = models[col]
        le = label_encoders[col]
        predicted = clf.predict_proba(X_test_scaled)
        num_classes = len(le.classes_)
        num_groups = len(predicted) // group_size

        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            class_probs = [sum([group_pred[k][j] for k in range(group_size)]) / group_size for j in range(num_classes)]
            
            unique_id = df_y_info.iloc[i * group_size]['unique_id']
            if unique_id not in final_results:
                final_results[unique_id] = {}

            if col in ['gender', 'hold racket handed']:
                final_results[unique_id][col] = round(class_probs[1], 4)  # 取 class_1 的機率
            else:
                for j, class_name in enumerate(le.classes_):
                    key = f"{col}_{class_name}"
                    final_results[unique_id][key] = round(class_probs[j], 4)

    # 轉換成 DataFrame 並輸出
    output_df = pd.DataFrame.from_dict(final_results, orient='index').reset_index()
    output_df = output_df.rename(columns={'index': 'unique_id'})

    # 確保欄位順序
    columns = ['unique_id', 'gender', 'hold racket handed',
               'play years_0', 'play years_1', 'play years_2',
               'level_2', 'level_3', 'level_4', 'level_5']
    for col in columns:
        if col not in output_df.columns:
            output_df[col] = np.nan

    output_df = output_df[columns]
    output_df.to_csv('test_predictions.csv', index=False)
    print("已儲存 test_predictions.csv")

def load_models_and_encoders(model_dir='saved_models'):
    models = {}
    label_encoders = {}
    for target in ['gender', 'hold racket handed', 'play years', 'level']:
        model_path = f"{model_dir}/model_{target}.joblib"
        le_path = f"{model_dir}/le_{target}.joblib"
        models[target] = joblib.load(model_path)
        label_encoders[target] = joblib.load(le_path)
    return models, label_encoders

def train_and_save_model(X, y, name, save_dir="saved_models"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=100,
        max_depth=6
    )
    model.fit(X, y_encoded)

    joblib.dump(model, os.path.join(save_dir, f"model_{name}.joblib"))
    joblib.dump(le, os.path.join(save_dir, f"le_{name}.joblib"))

    return model, le

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    # data_generate('./train_data', 'tabular_data_train')
    
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    
    # 標準化特徵
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    group_size = 27

    # 建立儲存資料夾
    os.makedirs("saved_models", exist_ok=True)

    models = {}
    label_encoders = {}
    for target in target_mask:
        models[target], label_encoders[target] = train_and_save_model(X_train_scaled, y_train[target], target)

    # 儲存 scaler
    joblib.dump(scaler, 'saved_models/scaler.pkl')

if __name__ == '__main__':
    main()
