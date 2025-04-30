import os, gzip, math
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
import core_functions as cf
from core_functions import config

# Вывод информации о плохом значении функции потерь и метрике
def print_loss_acc_info(n_samples: int) -> None:
    # Вычисление значения loss при работе модели наугад: 50%. 
    # Для 1 примера: ln(2) * (1+n_samples); -ln(0.5) = ln(2)
    # Для batch_size примеров: ln(2) * (1+n_samples) * batch_size / batch_size   
    bad_loss = math.log(2) * (1 + n_samples)

    # Информация
    cf.print_key_info('>ИНФОРМАЦИЯ<')
    cf.print_info('Модель работает наугад при значении loss=', end=''); cf.print_key_info(f'{bad_loss:.6f}')
    cf.print_info('В случае предобученной модели значение loss не "нативное", оно не является ориентиром.')
    cf.print_info('Метрика [ACCURACY] W2V/SG/NS - это поверхностная оценка вероятности предсказания.')
    cf.print_info('Правильные примеры:\t>50%')
    cf.print_info('Неправильные примеры:\t<50%')
    cf.print_info('Основная оценка метрики модели - это визуальная', end=' '); cf.print_key_info('оценка близких слов.', end='\n\n')

def print_model_info(model_name: str) -> None:
    try:
        info = api.info(model_name)
        if info:
            cf.print_info(f'ИНФОРМАЦИЯ О МОДЕЛИ', end=' '); cf.print_key_info(f'{model_name}')
            for key, value in info.items():
                cf.print_info(f'{key}: {value}')
            print()
        else:
            cf.print_warn('Информация о модели не найдена.')
            cf.print_warn(f'Модель: {model_name}', end='\n\n')
    except Exception as e:
        cf.print_error('Невозможно получить информацию о модели.')
        cf.print_error(f'Модель: {model_name}')
        cf.print_error(f'Причина: {e}', end='\n\n')
                
def check_and_load_model(pretrained_models_dir_path: str) -> KeyedVectors:
    if not os.path.isdir(pretrained_models_dir_path):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь: {pretrained_models_dir_path}', end='\n\n')
        exit(1)
    
    model_path = os.path.join(pretrained_models_dir_path, f'{config.w2v_model_filename}.model')
    
    if os.path.isfile(model_path):
        cf.print_info("Используется предобученная модель.")
        cf.print_info("Модель:", end=' '); cf.print_key_info(f'{config.w2v_model_filename}')
        cf.print_info("Путь:",end=' '); cf.print_key_info(f'{model_path}', end='\n\n')
    else:
        try:
            cf.print_info("Предобученная модель gensim скачивается.")
            cf.print_info("Модель:", end=' '); cf.print_key_info(f'{config.w2v_model_filename}')
            cf.print_info("Пожалуйста, подождите...")
            pretrained_model_path = api.load(config.w2v_model_filename, return_path=True)
            with gzip.open(pretrained_model_path, 'rb') as f_in:
                with open(f'{model_path}', 'wb') as f_out:
                    f_out.write(f_in.read())
            cf.print_info("Модель сохранена по пути:",end=' '); cf.print_key_info(f'{model_path}', end='\n\n')
        except Exception as e:
            cf.print_critical_error('Невозможно загрузить предобученную модель gensim.', prefix='\n')
            cf.print_critical_error(f'Модель: {config.w2v_model_filename}')
            cf.print_critical_error(f'Причина: {e}', end='\n\n')
            exit(1)
            
    print_model_info(config.w2v_model_filename)
    return KeyedVectors.load_word2vec_format(f'{model_path}', binary=True)

def get_vocab_and_embeddings(pretrained_w2v_model: KeyedVectors) -> tuple[dict[str, int], list[str],np.ndarray, int]:
    try:
        pretrained_word2idx = pretrained_w2v_model.key_to_index
        pretrained_idx2word = pretrained_w2v_model.index_to_key
        pretrained_embeddings = pretrained_w2v_model.vectors
        vector_size = pretrained_w2v_model.vector_size
        
        cf.print_info('Предобученные данные модели gensim были успешно получены.')
        cf.print_info('Модель:\t\t\t\t', end=''); cf.print_key_info(f'{config.w2v_model_filename}')
        cf.print_info('Размер словаря:\t\t\t', end=''); cf.print_key_info(f'{len(pretrained_word2idx)}')
        cf.print_info('Размер матрицы эмбеддингов:\t', end=''); cf.print_key_info(f'{pretrained_embeddings.shape}')
        cf.print_info('Размер вектора эмбеддинга:\t', end=''); cf.print_key_info(f'{vector_size}')
        
        return pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size
    except Exception as e:
        cf.print_critical_error('Невозможно получить данные предобученной модели gensim.', prefix='\n')
        cf.print_critical_error(f'Модель: {config.w2v_model_filename}')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)