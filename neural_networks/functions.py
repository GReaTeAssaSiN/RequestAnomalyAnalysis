from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import math
import core_functions as cf

# Разбиение данных на обучающую, валидационную и тестовую выборки.
def split_dataset(clean_requests: list[list[str]], val_size: float, test_size: float, seed: int=42) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    # Обработка аргументов
    total_size = len(clean_requests)
    
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1.0:
        raise ValueError("val_size и test_size должны быть неотрицательными, а их сумма — меньше 1.")

    if val_size == 0 and test_size == 0:
        cf.print_warn(f'Вся выборка будет использована как тестовая: {total_size} обращений')
        return [], [], clean_requests

    # Обучающая + валидационная/тестовая
    train_data, temp_data = train_test_split(clean_requests, test_size=val_size+test_size, random_state=seed, shuffle=True)
    # Валидационная + тестовая
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(val_size + test_size), random_state=seed, shuffle=True)
    cf.print_info('Корпус был разбит на обучающую, валидационную и тестовую выборки:')
    cf.print_info('Обучающая:\t', end=''); cf.print_key_info(f'{len(train_data)}')
    cf.print_info('Валидационная:\t', end=''); cf.print_key_info(f'{len(val_data)}')
    cf.print_info('Тестовая:\t', end=''); cf.print_key_info(f'{len(test_data)}')
    
    return train_data, val_data, test_data

# Получение устройства, на котором выполняются вычисления: CPU/GPU
def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf.print_info(f'Устройство, на котором будут производится вычисления:', end=' ')
    cf.print_key_info(f'{device}', end='\n\n')
    return device

""" WORD2VECTOR - SKIP-GRAM - NEGATIVE SAMPLING """
# Построение словаря и эмбеддингов по обучающему корпусу + предобученные параметры
def build_vocab_and_embeddings(train_data: list[list[str]], pretrained_word2idx: dict[str, int], pretrained_idx2word: list[str],
                               pretrained_embeddings: np.ndarray, vector_size: int, min_count=1) -> tuple[dict[str,int], list[str], torch.Tensor]:
    # Проверка
    if not (isinstance(pretrained_word2idx, dict) and isinstance(pretrained_idx2word, list) and isinstance(pretrained_embeddings, np.ndarray) and isinstance(vector_size, int) and vector_size>0):
        raise ValueError('Параметры pretrained_embeddings, pretrained_word2idx, pretrained_idx2word, vector_size переданы не корректно, vector_size должен быть > 0.\n')
    
    # Копирование предобученных параметров
    if pretrained_word2idx and pretrained_idx2word and pretrained_embeddings.size > 0:
        word2idx = dict(pretrained_word2idx)
        idx2word = list(pretrained_idx2word)
        embeddings = list(pretrained_embeddings)
        cf.print_info('Данные словаря и матрицы эмбеддингов предобученной модели gensim W2V/SG были скопированы.')
    else:
        word2idx = {}
        idx2word = []
        embeddings = []
        cf.print_info('Предобученные данные модели Word2Vec/SkipGram отсутствуют.')
        if not train_data:
            cf.print_critical_error('Словарь и матрица эмбеддингов не могут быть пустыми.', end='\n\n', prefix='\n')
            exit(1)
    
    # Дополнение/формирование словаря и матрицы эмбеддингов
    cf.print_info('Производится формирования словаря и матрицы эмбеддингов для модели W2V/SG/NS. Пожалуйста, подождите...')    
    if train_data:
        # Подсчет частоты всех токенов в обучающем корпусе
        token_freqs = Counter(token for request in train_data for token in request)
        # Добавление новых токенов, частота которых >= min_count
        for token, freq in token_freqs.items():
            if freq >= min_count and token not in word2idx:
                idx = len(word2idx)
                word2idx[token] = idx
                idx2word.append(token)
                embeddings.append(np.random.uniform(-1, 1, vector_size))
    # Привединие матрицы эмбеддингов к тензору: np.ndarray -> torch.Tensor
    embedding_matrix = torch.tensor(np.array(embeddings), dtype=torch.float32)
    
    cf.print_info('Словарь и матрица эмбеддингов были успешно сформированы.')
    cf.print_info('Размер словаря:\t\t\t', end='')
    cf.print_key_info(f'{len(word2idx)}')
    cf.print_info('Размер матрицы эмбеддингов:\t', end='')
    cf.print_key_info(f'{tuple(embedding_matrix.shape)}')
    cf.print_info('Размер вектора эмбеддинга:\t', end='')
    cf.print_key_info(f'{embedding_matrix.shape[1]}', end='\n\n')
    
    return word2idx, idx2word, embedding_matrix
 
# Генерация пар: (центральное слово, контекстное слово)
def generate_skipgram_data_pairs(tokenized_data: list[list[str]], word2idx: dict[str, int], window_size=10) -> list[tuple[int, int]]:
    cf.print_info('Производится генерация пар (центральное слово, контекстное слово) для W2V/SG/NS. Пожалуйста, подождите...')
    pairs = []
    
    if not tokenized_data:
        cf.print_critical_error('Пары не могут быть сгенерированы из пустых данных.', end='\n\n', prefix='\n')
        exit(1)

    for request in tokenized_data:
        indexed_request = [word2idx[word] for word in request if word in word2idx]
        
        for center_pos in range(len(indexed_request)):
            center_word = indexed_request[center_pos]

            # Контекстное окно
            for offset in range(-window_size, window_size + 1):
                context_pos = center_pos + offset
                if offset == 0 or context_pos < 0 or context_pos >= len(indexed_request):
                    continue
                context_word = indexed_request[context_pos]
                pairs.append((center_word, context_word))
    cf.print_info('Пары были сгенерированы! Количество:', end=' '); cf.print_key_info(f'{len(pairs)}')         
    return pairs

# Получение эмбеддингов слов словаря модели из самой модели
def get_trained_embeddings(model: nn.Module) -> np.ndarray:
    cf.print_info('Осуществляется обновление и получение эмбеддингов слов словаря из модели. Пожалуйста, подождите...')
    embedding_matrix = model.in_embeddings.weight.detach().cpu().numpy()
    cf.print_info('Эмбеддинги были обновлены и получены. Размер матрицы эмбеддингов', end=' '); cf.print_key_info(f'{embedding_matrix.shape}.')
    return embedding_matrix

# Вычисление TF-IDF слов словаря
def compute_tf_idf(corpus: list[list[str]], word2idx: dict[str, int]) -> dict[str, float]:
    cf.print_key_info('>TF-IDF вычисляется на основе всего корпуса<')
    cf.print_info('Осуществляется вычисление параметра TF-IDF для слов в словаре. Пожалуйста, подождите...')
    num_requests = len(corpus)               # Общее число обращений
    tf_dict = {word: 0 for word in word2idx} # Частота термина (TF) для каждого слова в словаре в рамках обращения
    df_dict = {word: 0 for word in word2idx} # Документная частота термина (DF) для каждого слова в словаре в рамках всех обращений
    
    for request in corpus:
        request_word_counts = Counter(request)  # Счетчик слов в обращении
        for word, count in request_word_counts.items():
            # Для каждого слова из обращения в словаре добавляем TF в словарь tf_dict
            if word in word2idx:
                tf_dict[word] += count / len(request)
    
        # Для каждого слова из обращения в словаре добавляем DF в словарь df_dict
        unique_words_in_request = set(request)
        for word in unique_words_in_request:
            if word in word2idx:
                df_dict[word] += 1
    
    # Вычисляем IDF для каждого слова в словаре
    idf_dict = {}
    zero_count = 0
    for word in word2idx:
        # IDF = ln(количество обращений/DF слова)
        if df_dict[word] > 0:
            idf_dict[word] = math.log(num_requests / df_dict[word])
        else:
            idf_dict[word] = 0
            zero_count += 1

    # TF-IDF = TF слова * IDF слова
    tfidf_dict = {word: tf_dict[word] * idf_dict[word] for word in word2idx}
    cf.print_info('TF-IDF значение для слов в словаре было вычислено.', end='\n\n')
    cf.print_info('Статистика:')
    cf.print_info('TF-IDF = 0 для слов в количестве:\t', end=''); cf.print_key_info(f'{zero_count}')
    cf.print_info('TF-IDF !=  0 для слов в количестве:\t', end=''); cf.print_key_info(f'{len(word2idx) - zero_count}', end='\n\n')
    return tfidf_dict

# Получение эмбеддингов обращений
def get_request_embeddings(corpus: list[list[str]], word2idx: dict[str, int], embeddings: np.ndarray, vector_size: int, tfidf_dict: dict[str, float]) -> np.ndarray:
    cf.print_info('Осуществляется получение эмбеддинга(-ов) обращения(-й). Пожалуйста, подождите...')
    
    request_embeddings = []
    unidentified_reqsts_count = 0
    for request in corpus:
        sum_embeddings = np.zeros(vector_size)
        num_words_in_request = 0
        for word in request:
            if word in word2idx:
                num_words_in_request += 1
                word_idx = word2idx[word]
                word_embedding = embeddings[word_idx]
                tfidf_weight = tfidf_dict.get(word, 0)

                # Эмбеддинг = смысл слова * важность
                weighted_embedding = word_embedding * tfidf_weight
                
                sum_embeddings += weighted_embedding
 
        if num_words_in_request > 0:
            avg_embedding = sum_embeddings / len(request)
        else:
            unidentified_reqsts_count += 1
            avg_embedding = np.zeros(vector_size)

        request_embeddings.append(avg_embedding)

    cf.print_info('Эмбеддинг(-и) обращения(-й) был(-и) получен(-ы).')
    cf.print_info('Количество неопознанных предложений:\t', end=''); cf.print_key_info(f'{unidentified_reqsts_count}')
    cf.print_info('Количество опознанных предложений:\t', end=''); cf.print_key_info(f'{len(corpus) - unidentified_reqsts_count}')
    return np.array(request_embeddings)