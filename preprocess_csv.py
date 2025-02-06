# 预处理csv文件，输出处理后的csv文件及对应的字典pickle文件
from sklearn.preprocessing import LabelEncoder,KBinsDiscretizer
from get_path import get_dict_path,get_decoder_path,get_bin_value_dict_path,get_unique_path
from dateutil import parser
import pandas as pd
import json
import pickle
import os

def discretize_column(column, k):
    """
    将数值列进行分桶处理，并创建一个字典，包含桶边界和对应的整数标签。

    参数:
    column (pd.Series): 要分桶的数值列。
    k (int): 桶的个数。

    返回:
    tuple: 包含分桶后的列和一个字典，字典的键是桶边界，值是对应的整数标签。
    """
    discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
    binned_column = discretizer.fit_transform(column.values.reshape(-1, 1)).flatten().astype(int)
    # 用法：discretizer.transform((1,1))得到bin_label
    return binned_column, discretizer
# 定义一个函数，尝试解析日期并返回布尔值，排除NaN
def is_date(date_str):
    if pd.isna(date_str):
        return True  # 或者根据你的需求返回False
    try:
        parser.parse(date_str)
        return True
    except:
        return False
def convert_to_datetime(series):
    datetime_series = pd.to_datetime(series)  # 将无法转换的设置为NaT
    return datetime_series
def date_preprocess(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path, escapechar='\\',low_memory=False)    
    # 遍历DataFrame的每一列，检测日期类型
    for column in df.columns:
        if not is_date(df[column][0]):
            continue
        if df[column].apply(is_date).all():
            df[column]=convert_to_datetime(df[column])
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            # 将日期类型的列转换为时间戳的整数
            df[column] = df[column].apply(lambda x: int(x.timestamp()) if pd.notnull(x) else x)
    return df

def csv_exclude_nan(csv_path,k):
    """
    预处理CSV文件，包括读取、编码非数值列、数值分桶、编码缺失值，并保存处理结果和编码对象。

    参数:
    csv_path (str): CSV文件的路径。
    k (int): 数值分桶的桶个数。

    返回:
    tuple: 包含预处理后的CSV文件路径和一个编码对象路径的元组。
    """
    df=date_preprocess(csv_path)
    csv_name = os.path.basename(csv_path).split('.')[0]
    pkl_dir = os.path.join(os.path.dirname(csv_path), f"{csv_name}_decoders")
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    pkl_dir = os.path.join(os.path.dirname(csv_path), f"{csv_name}_dicts")
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_nan_series = df[col].dropna()
            original_col=df[col].fillna('anan')
            unique_values=df[col].unique()
            # 对数值列进行分桶处理，并获取分桶字典
            binned_series, bin_decoder = discretize_column(non_nan_series, k)
            df.loc[df[col].notna(), col] = binned_series
            # 将 NaN 值替换为分桶个数 k
            df[col] = df[col].fillna(k)
            df[col] = df[col].astype(int)
            # decoder_path = os.path.join(pkl_dir, os.path.basename(f"{col}_bin_decoder.pkl"))
            compare_df=pd.concat([original_col.rename('ori'),df[col]],axis=1)
            bin_value_dict=compare_df.groupby(col).apply(lambda x: x['ori'].value_counts().to_dict(),include_groups=False).to_dict()
            # bin_value_dict=df[col].value_counts().to_dict()
            bin_dict_path=get_bin_value_dict_path(csv_path,col)
            unique_path=get_unique_path(csv_path,col)
            with open(bin_dict_path,"wb") as f:
                pickle.dump(bin_value_dict,f)
            decoder_path = get_decoder_path(csv_path,col)
            with open(decoder_path, "wb") as f:
                pickle.dump(bin_decoder, f)
            with open(unique_path,'wb') as f:
                pickle.dump(unique_values,f)
        else:
            df[col]=df[col].fillna('anan')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            df[col],bin_decoder = discretize_column(df[col], k)
            classes = le.classes_
            encoded_values = le.transform(classes)
            bin_labels = bin_decoder.transform(encoded_values.reshape(-1,1))
            encoding_dict = dict(zip(classes, bin_labels))
            dict_path=get_dict_path(csv_path,col)
            with open(dict_path, "wb") as f:
                pickle.dump(encoding_dict, f)
    save_path = csv_path.split('.')[0] + '_preprocessed.csv'
    df.to_csv(save_path, index=False)
    return save_path, pkl_dir

def csv_bin_process(csv_path,k):
    """
    预处理CSV文件，包括读取、填充缺失值、编码非数值列、数值分桶，并保存处理结果和编码对象。

    参数:
    csv_path (str): CSV文件的路径。
    k (int): 数值分桶的桶个数。

    返回:
    tuple: 包含预处理后的CSV文件路径和一个编码对象路径的元组。
    """
    # df = pd.read_csv(csv_path, escapechar='\\')
    df=date_preprocess(csv_path)
    csv_name = os.path.basename(csv_path).split('.')[0]
    # df = df.fillna('anan')
    pkl_dir = os.path.join(os.path.dirname(csv_path), f"{csv_name}_decoders")
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    pkl_dir = os.path.join(os.path.dirname(csv_path), f"{csv_name}_dicts")
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].fillna(-1)
            original_col=df[col].copy()
            unique_values=df[col].unique()
            # 对数值列进行分桶处理，并获取分桶字典
            df[col], bin_decoder = discretize_column(df[col], k)
            # decoder_path = os.path.join(pkl_dir, os.path.basename(f"{col}_bin_decoder.pkl"))
            compare_df=pd.concat([original_col.rename('ori'),df[col]],axis=1)
            bin_value_dict=compare_df.groupby(col).apply(lambda x: x['ori'].value_counts().to_dict(),include_groups=False).to_dict()
            # bin_value_dict=df[col].value_counts().to_dict()
            bin_dict_path=get_bin_value_dict_path(csv_path,col)
            unique_path=get_unique_path(csv_path,col)
            with open(bin_dict_path,"wb") as f:
                pickle.dump(bin_value_dict,f)
            decoder_path = get_decoder_path(csv_path,col)
            with open(decoder_path, "wb") as f:
                pickle.dump(bin_decoder, f)
            with open(unique_path,'wb') as f:
                pickle.dump(unique_values,f)
        else:
            df[col]=df[col].fillna('anan')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            df[col],bin_decoder = discretize_column(df[col], k)
            classes = le.classes_
            encoded_values = le.transform(classes)
            bin_labels = bin_decoder.transform(encoded_values.reshape(-1,1))
            encoding_dict = dict(zip(classes, bin_labels))
            # print(encoding_dict)
            # dict_path=os.path.join(pkl_dir,os.path.basename(f"{col}_dict.pkl"))
            dict_path=get_dict_path(csv_path,col)
            with open(dict_path, "wb") as f:
                pickle.dump(encoding_dict, f)
            # decoder_path = os.path.join(pkl_dir, os.path.basename(f"{col}_label_decoder.pkl"))
            # with open(decoder_path, "wb") as f:
                # pickle.dump(le, f)
            # decoder_path = os.path.join(pkl_dir, os.path.basename(f"{col}_bin_decoder.pkl"))
            # with open(decoder_path, "wb") as f:
                # pickle.dump(bin_decoder, f)
    save_path = csv_path.split('.')[0] + '_preprocessed.csv'
    df.to_csv(save_path, index=False)
    return save_path, pkl_dir

def csv_process(csv_path):
    """
    预处理CSV文件，包括读取、填充缺失值、编码非数值列，并保存处理结果。
    
    参数:
    csv_path (str): CSV文件的路径。
    
    返回:
    save_ptah: 预处理后的CSV文件路径
    """
    df=pd.read_csv(csv_path,escapechar='\\')
    csv_name=os.path.basename(csv_path).split('.')[0]
    # df=df.fillna('anan')
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].fillna(-1)
        else:
            df[col]=df[col].fillna('anan')
        if False:
            continue
        else:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])
    save_path=csv_path.split('.')[0]+'_preprocessed.csv'
    df.to_csv(save_path, index=False)
    return save_path

def csvpreprocess(csv_path):
    """
    预处理CSV文件，包括读取、填充缺失值、编码非数值列，并保存处理结果。
    
    参数:
    csv_path (str): CSV文件的路径。
    
    返回:
    tuple: 包含预处理后的CSV文件路径和一个编码字典路径的元组。
    """
    df=pd.read_csv(csv_path,escapechar='\\')
    csv_name=os.path.basename(csv_path).split('.')[0]
    # df=df.fillna('anan')
    pkl_dir=os.path.join(os.path.dirname(csv_path),os.path.basename(f"/{csv_name}_dicts"))
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].fillna(-1)
        else:
            df[col]=df[col].fillna('anan')
        if False:
            # count_dict={'count':len(df[col].unique())}
            continue
        else:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])
            classes = le.classes_
            encoded_values = le.transform(classes)
            encoding_dict = dict(zip(classes, encoded_values))
            # print(encoding_dict)
            dict_path=os.path.join(pkl_dir,os.path.basename(f"{col}_dict.pkl"))
            with open(dict_path, "wb") as f:
                pickle.dump(encoding_dict, f)
    save_path=csv_path.split('.')[0]+'_preprocessed.csv'
    df.to_csv(save_path, index=False)
    return save_path,os.path.dirname(dict_path)
# def get_dict_path(table_path,attr):
#     table_dir=os.path.dirname(table_path)
#     table_name=os.path.basename(table_path).split('.')[0]
#     return os.path.join(table_dir,f'{table_name}_dicts',f'{attr}_dict.pkl')
# def get_bin_value_dict_path(table_path,attr):
#     table_dir=os.path.dirname(table_path)
#     table_name=os.path.basename(table_path).split('.')[0]
#     return os.path.join(table_dir,f'{table_name}_decoders',f'{attr}_dict.pkl')
# def get_decoder_path(table_path,attr):
#     table_dir=os.path.dirname(table_path)
#     table_name=os.path.basename(table_path).split('.')[0]
#     return os.path.join(table_dir,f'{table_name}_decoders',f'{attr}_decoder.pkl')
# def get_dict_path(dict_dir,col_name):
    # return os.path.join(dict_dir,os.path.basename(f'{col_name}_dict.pkl'))

if __name__=="__main__":
    csv_exclude_nan('test_data/stu_table_tiny.csv',k=1000)
    # csv_bin_process('test_data/stu_table_tiny.csv',k=1000)
    # csvpreprocess('test_data/stu_table_tiny.csv')