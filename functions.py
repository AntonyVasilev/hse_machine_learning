import streamlit as st # type: ignore
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures


poly_cols = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
ohe_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']


def prepare_loaded_dataframe(
    data: pd.DataFrame, 
    cols_with_na: list, 
    na_fill_mapping: dict,
    feature_names: list,
    columns: list,
    final_columns: list,
    min_max_scaler: MinMaxScaler, 
    one_hot_encoder: OneHotEncoder, 
    ordinal_encoder: OrdinalEncoder, 
    polynominal_features: PolynomialFeatures
    ) -> pd.DataFrame:
    """Применение всех необходимых преобразований к датафрейму, на котором будет делаться предсказание

    Args:
        data (pd.DataFrame): Исходный датафрейм, который загрузил пользователь

    Returns:
        pd.DataFrame: Преобразованный датафрейм, готовый к использованию в модели
    """
    df = data.copy()

    # Удаление дубликатов в данных
    idxs_to_drop = df.drop('selling_price', axis=1).duplicated(keep='first')
    df = df.loc[~idxs_to_drop, :].reset_index(drop=True)

    # Приведение признаков 'mileage', 'engine', 'max_power' к формату float
    for col_ in ['mileage', 'engine', 'max_power']:
        df[col_] = df[col_].apply(convert_to_float)

    # Обработка значений признака torque - парсинг числовых значений
    cleaned_torque = df['torque'].apply(parce_torque)
    cleaned_torque_df = pd.DataFrame(cleaned_torque.to_list(), columns=['torque', 'max_torque_rpm'])
    df = pd.concat([df.drop('torque', axis=1), cleaned_torque_df], axis=1)

    # Сохраняю маски полей, которые были с пропущенными значениями, по каждому из заполняемых признаков 
    na_values_mask = {col: df[col].isna() for col in cols_with_na}

    # Заполнение медианой пропущенных значений
    df = df.fillna(na_fill_mapping)

    # Привожу значения seatsк целочисленному типу
    df['seats'] = df['seats'].astype('int')

    # Проделываю аналогичные преобразования с тестовыми данными
    df_prepared = df.select_dtypes(include='object')
    df_prepared = pd.concat([df_prepared, df[columns]], axis=1)
    df_prepared['name'] = df_prepared['name'].apply(lambda val: val.split()[0].strip())
    df_prepared['name'] = ordinal_encoder.transform(df_prepared[['name']])
    df_prepared['name'] = df_prepared['name'].astype('int')
    df_prepared = pd.concat([
        df_prepared,
        pd.DataFrame(data=polynominal_features.transform(df_prepared[poly_cols]), 
                     columns=polynominal_features.get_feature_names_out()).drop(poly_cols + ['1'], axis=1)
    ], axis=1)
    encoded_cols_test = pd.DataFrame(data=one_hot_encoder.fit_transform(df_prepared[ohe_cols]).toarray(), 
                                     columns=one_hot_encoder.get_feature_names_out())
    df_prepared = pd.concat([df_prepared.drop(ohe_cols, axis=1), encoded_cols_test], axis=1)

    # Добавляю недостающие в тесте столбцы
    cols_to_add = list(set(feature_names) - set(df_prepared.columns))
    df_prepared[cols_to_add] = 0
    df_prepared = df_prepared[final_columns]

    # Маштабирую данные с помощью MinMaxScaler
    df_prepared = pd.DataFrame(data=min_max_scaler.transform(df_prepared), columns=df_prepared.columns)

    for col, mask in na_values_mask.items():
        df_prepared[f"is_nan_val_{col}"] = mask.astype('int')
    
    return df_prepared


@st.cache_resource
def load_transforms(path: Path) -> tuple:
    """Загрузка энкодеров и скалеров для подготовки загруженного датафрейма / данных, введенных в форму

    Args:
        path (Path): Путь к папке с моделями 

    Returns:
        tuple: Кортеж с загруженными обученными классами энкодеров и скалеров
    """
    # min_max_scaler
    with open(path / 'min_max_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)

    # one_hot_encoder
    with open(path / 'one_hot_encoder.pkl', 'rb') as f: 
        one_hot_encoder = pickle.load(f)

    # ordinal_encoder
    with open(path / 'ordinal_encoder.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)

    # polynominal_features
    with open(path / 'polynominal_features.pkl', 'rb') as f:
        polynominal_features = pickle.load(f)

    return min_max_scaler, one_hot_encoder, ordinal_encoder, polynominal_features


@st.cache_resource
def load_additional_data(path: Path) -> tuple:
    """Загрузка вспомогательных данных, необходимых для корректной работы приложения

    Args:
        path (Path): Путь к папке с данными

    Returns:
        tuple: Кортеж с загруженными данными
    """
    # cols_with_na
    with open(path / 'cols_with_na.json', 'rb') as f:
        cols_with_na = json.load(f)

    # columns
    with open(path / 'columns.json', 'rb') as f:
        columns = json.load(f)

    # features_base_dtypes
    with open(path / 'features_base_dtypes.pkl', 'rb') as f:
        features_base_dtypes = pickle.load(f)

    # final_columns
    with open(path / 'final_columns.json', 'rb') as f:
        final_columns = json.load(f)

    # main_columns
    with open(path / 'main_columns.json', 'rb') as f:
        main_columns = json.load(f)
    
    # na_fill_mapping
    with open(path / 'na_fill_mapping.json', 'rb') as f:
        na_fill_mapping = json.load(f)
    
    # values_for_form
    with open(path / 'values_for_form.json', 'rb') as f:
        values_for_form = json.load(f)
    
    return cols_with_na, na_fill_mapping, columns, final_columns, main_columns, values_for_form, features_base_dtypes


def load_train_data(path: Path) -> pd.DataFrame:
    """Загрузка данных, использованных при обучении модели

    Args:
        path (Path): Путь к папке с данными

    Returns:
        pd.DataFrame: Датафрейм с загруженными данными
    """
    data = pd.read_parquet(path / 'df_train.parquet')
    return data


@st.cache_resource
def load_model(model_path, feacture_names_path):
    """Загружаем модель через pickle"""

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feacture_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]


def convert_to_float(val: str) -> float:
    """Приведение признаков 'mileage', 'engine', 'max_power' к формату float

    Args:
        val (str): Строковое значение

    Returns:
        float: Преобразованное значение
    """
    if val is not np.nan:
        splitted_val = val.split()
        val_to_convert = splitted_val[0] if splitted_val else '0'
        if not val_to_convert.isalpha():
            return float(val_to_convert)
        else:
            return 0.0


def convert_to_float_torque(val: str) -> float:
    """Корректная конвертация значений в типу float с учетом значений с разделителями порядков из точек и запятых 

    Args:
        val (str): Строковое значение

    Returns:
        float: Преобразованное значение
    """
    if ',' in val:
        val = ''.join(val.split(','))
    elif '.' in val:
        val = ''.join(val.split('.'))
    return float(val)


def parce_2_part(value: str) -> float:
    """Функция для корректного преобразования второй части значения torque, которое в итоге быдет записано в max_torque_rpm

    Args:
        value (str): Строковое значение

    Returns:
        float: Преобразованное значение
    """
    if '+' in value:
            splitted_values = value.strip().split('+')
            splitted_val_1 = convert_to_float_torque(splitted_values[0])
            splitted_val_2 = convert_to_float_torque(splitted_values[1].split('rpm')[0].strip().split('-')[1].strip())
            max_torque_rpm = splitted_val_1 + splitted_val_2
    elif '-' in value:
            max_torque_rpm = max([convert_to_float_torque(val.strip())
                for val in value.lower().split('rpm')[0].split('-')
                if convert_to_float_torque(val.strip())])
    elif '~' in value:
        max_torque_rpm = max([convert_to_float_torque(val.strip())
            for val in value.lower().split('rpm')[0].split('~')
            if convert_to_float_torque(val.strip())])
    else:
        max_torque_rpm = convert_to_float_torque(
            value.lower().split('rpm')[0].strip())
    return max_torque_rpm


def parce_torque(value: str) -> tuple[float, float]:
    """Функция для корректного парсинга значений из признака torque и разделения их на 2: torque и max_torque_rpm

    Args:
        value (str): Строковое значение для преобразования

    Raises:
        ValueError: В данных отсутствуют допустимые разделители

    Returns:
        tuple[float, float]: Кортеж с 2 значениями типа float для новых признаков: torque и max_torque_rpm
    """
    if (value is None) | (value is np.nan) | (value == ''):
        return None, None

    try:
        # Обрабатываю разные варианты разделителей
        if '@' in value:
            split_sign = '@'
        elif 'at' in value:
            split_sign = 'at'
        elif '/' in value:
            split_sign = '/'
        else:
            if 'nm' in value.lower():
                return convert_to_float_torque(value.lower().split('nm')[0].strip()), None
            else:
                raise ValueError(f'Unknoun split sign: {value}')

        splitted_value = value.split(split_sign)
        # Обрабатываю 1 часть строки:
        value_1 = splitted_value[0]
        if 'nm' in value_1.lower():
            torque = convert_to_float_torque(value_1.lower().split('nm')[0].strip())
        elif 'kgm' in value_1.lower():
            torque = convert_to_float_torque(value_1.lower().split('kgm')[0].strip()) * 9.80665
        elif '(' in value_1:
            torque = convert_to_float_torque(value_1.split('(')[0].strip())
        else:
            torque = convert_to_float_torque(value_1.strip()) * 9.80665

        # Обрабатываю 2 часть строки:
        value_2 = splitted_value[1:]
        if len(value_2) > 1:
            value_2 = value_2[0].lower().split('(')[0].strip()
            max_torque_rpm = parce_2_part(value_2)
        else:
            value_2 = value_2[0]
        max_torque_rpm = parce_2_part(value_2)
    except Exception as e:
        print(e)
        print(value)
    else:
        return torque, max_torque_rpm
