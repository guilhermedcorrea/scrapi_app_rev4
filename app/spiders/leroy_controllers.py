from typing import Any
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from scipy.stats import norm
from functools import wraps
import redis
import numpy as np
from scipy.stats import norm
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from typing import Any
import pandas as pd
from textblob import TextBlob  
from unidecode import unidecode


def serialize_to_json(data):
    try:
        serialized_data = json.dumps(data)
        return serialized_data
    except Exception as e:
        print("Erro ao serializar os dados para JSON:", e)
        return None

def create_redis_connection():
    return redis.Redis(host='localhost', port=6379, db=0)


def calculate_probability_of_sale(row):
    price_difference = row['sugestao_preco'] - row['precoconcorrente']
    std_deviation = row['preco_real'] - row['sugestao_preco']
    if std_deviation == 0:
        return 'N/A'
    z_score = price_difference / std_deviation
    probability = 1 - norm.cdf(z_score)  
    return '{:.2%}'.format(probability)


def build_generator(latent_dim, output_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(output_dim, activation='tanh')(x)
    generator = Model(inputs, x)
    return generator


def integrate_gan_data(redis_connection, generator, num_samples):
    synthetic_data = generate_synthetic_data(generator, num_samples)
    if redis_connection:
        store_data_in_redis(redis_connection, synthetic_data)

def store_data_in_redis(redis_connection, data):
    if redis_connection:
        try:
            serialized_data = serialize_to_json(data)
            redis_connection.set("data_key", serialized_data)
            print("Dados armazenados no Redis com sucesso.")
        except Exception as e:
            print("Erro ao armazenar os dados no Redis:", e)

def query_data_from_redis(redis_host, redis_port, redis_db, key):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        data = r.get(key)
        if data:
            data_decoded = data.decode("utf-8")
            data_df = pd.read_json(data_decoded, orient="records")
            return data_df
        else:
            print("Chave não encontrada no Redis.")
            return None
    except Exception as e:
        print("Ocorreu um erro ao consultar dados do Redis:", e)

def create_redis_connection():
    return redis.Redis(host='localhost', port=6379, db=0)

def calculate_probability_of_sale(row):
    price_difference = row['sugestao_preco'] - row['precoconcorrente']
    std_deviation = row['preco_real'] - row['sugestao_preco']
    if std_deviation == 0:
        return 'N/A'
    z_score = price_difference / std_deviation
    probability = 1 - norm.cdf(z_score)  
    return '{:.2%}'.format(probability)

def get_combined_data(redis_connection):
    data = query_data_from_redis(redis_connection.host, redis_connection.port, redis_connection.db, "data_key")
    
    if not data:
         query = """
            WITH CombinedData AS (
                SELECT 
                    t.[cod_treino], t.[titulo], t.[categoria], t.[cor], t.[tamanho], t.[marca],
                    t.[referencia], t.[voltagem], t.[ambiente], t.[subcategoria], t.[linha], 
                    t.[tipo], t.[imagem], t.[Caracteristica],
                    g.[cod_google], g.[url_google], g.[concorrente], g.[url_produto], 
                    g.[cod_produto], g.[ean], g.[preco], g.[custo], g.[margem], g.[statuspreco], 
                    g.[vendido], g.[utimavenda], g.[precoconcorrente], g.[nomeproduto], 
                    g.[categoria], g.[marca], g.[diferencaconcorrente],
                    p.[cod_produto], p.[nome], p.[marca], p.[preco], p.[quantidade], 
                    p.[imagem], p.[descricao], p.[atributos], p.[ean], p.[precodetalhes], 
                    p.[classificacao], p.[sugestaoals], p.[precosugerido], p.[concorrente], 
                    p.[precoconcorrente], p.[urlgoogle], p.[urlanuncio]
                FROM [DEV].[dbo].[treino_produtos] t
                FULL JOIN [DEV].[comercial].[google_shopping] g ON t.[cod_treino] = g.[cod_produto]
                FULL JOIN [DEV].[comercial].[produtos] p ON t.[cod_treino] = p.[cod_produto]
            )
            SELECT * FROM CombinedData
            """
        
    
    return data


def generate_synthetic_data(generator, num_samples):
    latent_dim = generator.input_shape[1] 
    synthetic_data = generator.predict(np.random.normal(0, 1, (num_samples, latent_dim)))
    return synthetic_data


def integrate_gan_data(redis_connection, generator, num_samples):
    synthetic_data = generate_synthetic_data(generator, num_samples)
    
   
    if redis_connection:
        store_data_in_redis(redis_connection, synthetic_data)


def train_and_predict_from_data(data):
    dados = data 
    
    dados.fillna(0, inplace=True)
    dados[['preco', 'margem', 'precoconcorrente']] = dados[['preco', 'margem', 'precoconcorrente']].applymap(
        lambda k: float(str(k).replace(",", "").replace(".", "")))

    object_columns = dados.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(dados[object_columns])
    encoded_columns = encoder.get_feature_names_out(object_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    dados_encoded = pd.concat([dados.drop(object_columns, axis=1), encoded_df], axis=1)

    X = dados_encoded.drop("preco", axis=1)
    y = dados["preco"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    regressor = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(regressor, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_regressor = grid_search.best_estimator_

    results_df = pd.DataFrame({
        'nomeproduto': dados.loc[X_test.index, 'nomeproduto'], 
        'precoconcorrente': X_test['precoconcorrente'],
        'preco_real': y_test,
        'sugestao_preco': best_regressor.predict(X_test)
    })

    results_df['diferenca_preco'] = results_df['sugestao_preco'] - results_df['precoconcorrente']

    return results_df


def analyze_and_suggest_prices(redis_connection):
    def decorator(func):
        @wraps(func)
        def wrapper(driver, *args, **kwargs):
            result = func(driver, *args, **kwargs)
            data = get_combined_data(redis_connection)
            
            results_df = train_and_predict_from_data(data)
            
          
            results_df['sugestao_preco_analisado'] = results_df['sugestao_preco'] + 10  
            results_df['diferenca_preco_analisado'] = results_df['sugestao_preco_analisado'] - results_df['precoconcorrente']
            
         
            results_df['sugestao_preco_analisado'] = results_df.apply(
                lambda row: calculate_suggested_price(row), 
                axis=1
            )
            
            return results_df
        return wrapper
    return decorator


def calculate_suggested_price(row):
    venda_media = row['venda_media'] 
    preco_leroy = row['preco_leroy'] 
    custo = row['custo']  
    concorrentes = row['concorrentes']  
    
    
    sugestao_preco = (venda_media * 0.1) + preco_leroy + (concorrentes * 5) - (custo * 0.2)
    
    return sugestao_preco


def store_training_data_in_redis(redis_client):
    def decorator(func):
        def wrapper(*args, **kwargs):
            training_data = func(*args, **kwargs)
            for example in training_data:
                title_text, category, subcategory, cor, tamanho, marca, referencia, voltagem, ambiente, subcategoria, linha, tipo, imagem, caracteristica = example
               
                data_dict = {
                    "category": category,
                    "subcategory": subcategory,
                    "cor": cor,
                    "tamanho": tamanho,
                    "marca": marca,
                    "referencia": referencia,
                    "voltagem": voltagem,
                    "ambiente": ambiente,
                    "subcategoria": subcategoria,
                    "linha": linha,
                    "tipo": tipo,
                    "imagem": imagem,
                    "caracteristica": caracteristica
                }
                redis_client.hmset(title_text, data_dict)
               
                print("Dados armazenados no Redis para:", title_text)
            return training_data
        return wrapper
    return decorator


def process_title(driver, element):
    title_text = element.text
    

    title_text = title_text.upper()
    

    title_text = ' '.join(title_text.split())
    
    
    blob = TextBlob(title_text)
    corrected_text = str(blob.correct())
    
 
    corrected_text = corrected_text.replace('LEROY MERLIN', 'OUTRA LOJA')
    
    stored_data = redis_client.hgetall(corrected_text)
    if stored_data:
        category = stored_data.get(b'category', b'N/A').decode('utf-8')
        subcategory = stored_data.get(b'subcategory', b'N/A').decode('utf-8')
        cor = stored_data.get(b'cor', b'N/A').decode('utf-8')
        tamanho = stored_data.get(b'tamanho', b'N/A').decode('utf-8')
        marca = stored_data.get(b'marca', b'N/A').decode('utf-8')
        referencia = stored_data.get(b'referencia', b'N/A').decode('utf-8')
        voltagem = stored_data.get(b'voltagem', b'N/A').decode('utf-8')
        ambiente = stored_data.get(b'ambiente', b'N/A').decode('utf-8')
        subcategoria = stored_data.get(b'subcategoria', b'N/A').decode('utf-8')
        linha = stored_data.get(b'linha', b'N/A').decode('utf-8')
        tipo = stored_data.get(b'tipo', b'N/A').decode('utf-8')
        imagem = stored_data.get(b'imagem', b'N/A').decode('utf-8')
        caracteristica = stored_data.get(b'caracteristica', b'N/A').decode('utf-8')
        
       
        print("Categoria:", category)
        print("Subcategoria:", subcategory)
       
    else:
       
        print("Informações não encontradas no Redis para:", corrected_text)
    
    
    print("Título processado:", corrected_text)

@store_training_data_in_redis(redis_client)
def select_training_data():
    training_data = [
        ("PISO LAMINADO CLICK NATURE 136X21,7CM M² ARTENS", "Piso Laminado", "Nature", "N/A", "136x21,7", "N/A", "N/A", "N/A", "N/A", "Click", "NULL", "NULL", "NULL"),
        
    ]
    return training_data

@celery.task
def process_title_task(driver, element):
    process_title(driver, element)

def classify_elements(driver: Any, model: Any, tokenizer: Any, label_encoder: Any) -> list:
 
    num_samples = 100 
    latent_dim = model.input_shape[1]  
    
    
    synthetic_data = generate_synthetic_data(model, num_samples)
    
  
    synthetic_data_tokens = tokenizer.texts_to_sequences(synthetic_data)
    synthetic_data_padded = pad_sequences(synthetic_data_tokens, maxlen=max_sequence_length, padding='post')
  
    predicted_labels = model.predict(synthetic_data_padded)
    
   
    decoded_labels = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
    
    classified_elements = decoded_labels.tolist()
    
    return classified_elements


def process_page_decorator(func):
    def wrapper(driver, model, tokenizer, label_encoder):
        sql_query_and_store_in_redis()  
        
        elements = classify_elements(driver, model, tokenizer, label_encoder)
        for element in elements:
            element_type = element.get_attribute("data-type")
            if element_type == "descrição":
                process_description(element)
            elif element_type == "atributo":
                process_attribute(element)
            elif element_type == "preço":
                process_price(element)
                wait_for_page_to_reload(driver, driver.current_url)  
                enter_and_search_zipcode(driver, "12345-678") 
                shipping_info = extract_shipping_info(driver) 
                print("Informações de frete:", shipping_info)
                simulate_shipping_times(driver, shipping_info)
        
        func(driver, model, tokenizer, label_encoder)
    return wrapper
