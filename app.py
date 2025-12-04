import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from pathlib import Path
import phik  # noqa: F401
from datetime import datetime
import traceback
from functions import load_model, prepare_features, load_transforms, prepare_loaded_dataframe, load_additional_data, load_train_data

st.set_page_config(page_title="Cars' selling price prediction", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ridge_model.pkl"

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.pkl"

target_col_name = 'selling_price'

#--------------------------------------------------------------------------------------------------
# Загружаем модель

try:
    model, feature_names = load_model(model_path=MODEL_PATH, feacture_names_path=FEATURE_NAMES_PATH)
    feature_names = list(feature_names)
    feature_names_with_target = feature_names.copy()
    feature_names_with_target.append(target_col_name)
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

try:
    min_max_scaler, one_hot_encoder, ordinal_encoder, polynominal_features = load_transforms(path=MODEL_DIR)
    cols_with_na, na_fill_mapping, columns, final_columns, main_columns, values_for_form, features_base_dtypes = load_additional_data(path=DATA_DIR)
except Exception as e:
    st.error(f"Ошибка загрузки для подготовки данных: {e}")
    st.stop()

#--------------------------------------------------------------------------------------------------
# Визуализация весов модели
st.subheader("Визуализация весов модели")

chkbx = st.checkbox(label='Добавить интерсепт')

weights = model.coef_
df_to_plot = pd.DataFrame(data=pd.concat([
    pd.Series(feature_names), 
    pd.Series(weights)
    ], axis=1)
)
df_to_plot.columns = ['feature_name', 'weight']

if chkbx:
    intercept_df = pd.DataFrame([{'feature_name': 'intercept', 'weight': model.intercept_}])
    df_to_plot = pd.concat([df_to_plot, intercept_df], ignore_index=True)

st.line_chart(df_to_plot, x='feature_name' ,y='weight', x_label='Weight', y_label='Feature names')


#--------------------------------------------------------------------------------------------------
# Визуализации данных
st.subheader("Визуализации обучающих данных")

# Загрузка данных из файла
df_to_plot = load_train_data(path=DATA_DIR)
df_to_plot['brand'] = df_to_plot['name'].apply(lambda val: val.split()[0])

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
# pie chart признака brand
    n_cars_by_brand = (
        df_to_plot.groupby('brand').agg(n_cars=('name', 'count')).reset_index().sort_values('n_cars'))
    n_cars_by_brand['n_cars_part'] = n_cars_by_brand['n_cars'] / n_cars_by_brand['n_cars'].sum()
    n_cars_by_brand.loc[n_cars_by_brand['n_cars_part'] < 0.02, 'brand'] = 'Other'
    fig1 = px.pie(
        n_cars_by_brand,
        values='n_cars',
        names='brand',
        title="Количество автомобилей по брендам"
    )
    st.plotly_chart(fig1, width='stretch')
with plot_col2:
    # Гистограмма распрелделения целевой переменной
    fig2 = px.histogram(
        df_to_plot, 
        x=target_col_name, 
        marginal="box", 
        title='Гистаграмма распрелделения целевой переменной'
    )
    st.plotly_chart(fig2, width='stretch')

plot_col3, plot_col4 = st.columns(2)

with plot_col3:
    # Phik-матрица корреляций признаков
    corr_matrix = df_to_plot.phik_matrix()
    fig3 = px.imshow(
        corr_matrix, 
        text_auto=True, 
        color_continuous_scale='RdBu', 
        title='Phik-матрица корреляций признаков'
    )
    st.plotly_chart(fig3, width='stretch')
with plot_col4:
    # Средние цены в по годам с учетом количества сидений в автомобиле
    mean_prices_by_year = df_to_plot.groupby(['year', 'seats']).agg(mean_price=(target_col_name, 'mean')).reset_index()
    mean_prices_by_year['seats'] = mean_prices_by_year['seats'].astype('int')
    fig4 = px.bar(
        mean_prices_by_year, 
        x='year', 
        y='mean_price', 
        color='seats', 
        color_continuous_scale='RdBu',
        title='Средние цены в по годам с учетом количества сидений в автомобиле'
    )
    st.plotly_chart(fig4, width='stretch')

#--------------------------------------------------------------------------------------------------
st.subheader("Получение прогноза стоимости")

# Выбор варианта получения прогноза:
# - 'Загрузка csv-файла' - пользователь загружает csv-файл с данными. Модель делает предсказание по каждому объекту из файла,
#    строит графики с визуализациями данных и рассчитывает метрики кажества предсказания
# - 'Ввод данных в форму' - Пользователь вводит данных в форму для предсказания одного объекта. Модель делает предсказание
#    и выводит получившийся прогноз

pred_type = st.radio(
    'Выберите тип получения прогноза',
    ['Загрузка csv-файла', 'Ввод данных в форму'],
    captions=[
        'Предсказание по всем объектам из файла + графики',
        'Предсказание по одному объекту на данных из формы'
    ]
)

#--------------------------------------------------------------------------------------------------
# В зависимости от выбранного выполняем необходимые действия
if pred_type == 'Загрузка csv-файла':
    # Загрузка CSV файла
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

    if uploaded_file is None:
        st.info("Загрузите CSV файл для начала работы")
        st.stop()

    # Загружаем данные
    df = pd.read_csv(uploaded_file)

    #--------------------------------------------------------------------------------------------------
    # Приведение данных, загруженных пользователем, к формату, требуемому моделью
    df_prepared = prepare_loaded_dataframe(
        data=df,
        cols_with_na=cols_with_na,
        na_fill_mapping=na_fill_mapping,
        feature_names=feature_names_with_target,
        columns=columns,
        final_columns=final_columns,
        min_max_scaler=min_max_scaler,
        one_hot_encoder=one_hot_encoder,
        ordinal_encoder=ordinal_encoder,
        polynominal_features=polynominal_features
    )

    #--------------------------------------------------------------------------------------------------
    # Получение предсказания
    try:
        df_prepared = prepare_features(df_prepared, feature_names)
        X, y = df_prepared, df[target_col_name]
        prediction = model.predict(X)
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")
        st.error(traceback.format_exc())
        st.stop()
        
    st.subheader("Метрики качества предсказания")

    #--------------------------------------------------------------------------------------------------
    # Метрики
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        r2 = r2_score(y, prediction)
        st.metric("R2", f"{r2:.2f}", help='Коэффицент детерминации R-квадрат')
    with metric_col2:
        rmse = mean_absolute_percentage_error(y, prediction) * 100
        st.metric("MAPE", f"{rmse:.2f}%", help='Средняя абсолютная процентная ошибка')

    prediction_df = pd.DataFrame(data=prediction, columns=['prediction'])
    st.download_button(
        label='Скачать результат',
        data=prediction_df.to_csv(index=False).encode("utf-8"),
        file_name=f"car_price_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        help='Скачать результат предсказания стоимости автомобилей в формате csv',
        mime="text/csv",
        icon=":material/download:",
    )
#--------------------------------------------------------------------------------------------------
else:
    st.subheader("Сделать предсказание для нового клиента")

    # Формирование полей формы для выбора/ввода значений
    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)
        input_data = {}
        
        with col_left:
            st.write("**Категориальные признаки:**")
            for col in main_columns:
                if features_base_dtypes[col] in ('object', 'bool'):
                    # unique_vals = sorted(df[col].astype(str).unique().tolist())
                    input_data[col] = st.selectbox(col, values_for_form[col], key=f"cat_{col}")
        
        with col_right:
            st.write("**Числовые признаки:**")
            for col in main_columns:
                if features_base_dtypes[col] not in ('object', 'bool'):
                    # val = df[col].median()                
                    input_data[col] = st.number_input(col, value=values_for_form[col], key=f"num_{col}")

        submitted = st.form_submit_button("Получить предсказание", width='stretch')

    #--------------------------------------------------------------------------------------------------
    # Когда назата кнопка "Получить предсказание" делаем предсказание
    if submitted:
        result = st.empty()
        try:
            # Преобразуем данные из формы в датафрейм и прифодим их к нужному для модели формату
            input_df = pd.DataFrame([input_data])
            input_df_prepared = prepare_loaded_dataframe(
                data=input_df,
                cols_with_na=cols_with_na,
                na_fill_mapping=na_fill_mapping,
                feature_names=feature_names_with_target,
                columns=columns,
                final_columns=final_columns,
                min_max_scaler=min_max_scaler,
                one_hot_encoder=one_hot_encoder,
                ordinal_encoder=ordinal_encoder,
                polynominal_features=polynominal_features
            )
            input_df_prepared = prepare_features(input_df_prepared, feature_names)
            # Расчет предсказания
            prediction = model.predict(input_df_prepared)
            prediction = round(float(prediction[0]), 2)

            result.success(f"**Предсказанная стоимость автомобиля:** {prediction}")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.error(traceback.format_exc())
