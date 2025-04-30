from typing import Optional
import numpy as np
import core_functions as cf
import neural_networks as nn
from preprocessing.functions import process_single_request
from . import config
from .utils import DirectoriesManager, CSVManager
from preprocessing import load_udpipe_model, process_requests
import os, torch
from sklearn.metrics.pairwise import cosine_similarity
from ufal.udpipe import Model
from collections import Counter

""" УТИЛИТЫ """
def get_consumer_requests(csv_manager: CSVManager, SOURCE_DATA_DIR_PATH: str, SOURCE_DATA_CSV_DIR_PATH: str) -> list[str]:
    """ ЧТЕНИЕ ОБРАЩЕНИЙ ПОТРЕБИТЕЛЕЙ """
    # Надпись
    cf.print_inscription('ЧТЕНИЕ ОБРАЩЕНИЙ ПОТРЕБИТЕЛЕЙ')
    # Процесс
    requests = csv_manager.read_source_data_from_csv(SOURCE_DATA_CSV_DIR_PATH)
    if requests is None:
        requests = csv_manager.load_and_save_source_data_to_csv(SOURCE_DATA_DIR_PATH, SOURCE_DATA_CSV_DIR_PATH)
    else:
        cf.print_info('Исходные данные были загружены из .csv файла.')
        cf.print_info('Путь к файлу:', end=' '); cf.print_key_info(f'{os.path.join(SOURCE_DATA_DIR_PATH, config.file_name)} -> {config.file_column}')
    # Разделительная черта
    cf.print_sub_line()
    
    return requests

def get_preprocessed_requests(csv_manager: CSVManager, requests: list[str], PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH: str, PRETRAINED_MODELS_DIR_PATH: str, mode: str='dev', load_model: bool = True) -> list[list[str]]:
    """ ПРЕДОБРАБОТКА ВХОДНЫХ ДАННЫХ  """
    # Надпись
    cf.print_inscription('ПРЕДОБРАБОТКА ВХОДНЫХ ДАННЫХ')
    # Процесс
    preprocessed_requests = csv_manager.read_processed_data_from_csv(PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, suffix=None)
    if preprocessed_requests is None:
        if load_model: model = load_udpipe_model(PRETRAINED_MODELS_DIR_PATH)
        preprocessed_requests = process_requests(requests, csv_manager, PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, model, mode)
        csv_manager.save_processed_data_to_csv(preprocessed_requests, PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, suffix=None)
    else:
        cf.print_info('Полная статистика предобработки находится в сохраненном файле.')
        file_path = os.path.join(PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, f'{os.path.splitext(csv_manager.file_name)[0]}_{csv_manager.file_column}_stats.json')
        cf.print_info(f'Путь к файлу:', end=' '); cf.print_key_info(f'{file_path}')
    # Разделительная черта
    cf.print_sub_line()
    
    return preprocessed_requests

def get_cleaned_requests_from_None(preprocessed_requests: list[list[str]]) -> list[list[str]]:
    """ ОЧИСТКА ПРЕДОБРАБОТАННЫХ ДАННЫХ """
    # Надпись
    cf.print_inscription('ОЧИСТКА ПРЕДОБРАБОТАННЫХ ДАННЫХ') 
    # Процессы
    cf.print_info('Осуществляется чистка предобработанных обращений из .csv файла. Пожалуйста, подождите...')
    clean_preprocessed_requests = []
    for request in preprocessed_requests:
        clean_request = [token for token in request if token is not None]
        clean_preprocessed_requests.append(clean_request)
    cf.print_info('Предобработанные обращение были почищены.')
    # Разделительная черта
    cf.print_sub_line()
    
    return clean_preprocessed_requests

def load_and_get_pretrained_w2v_model(PRETRAINED_MODELS_DIR_PATH: str) -> tuple[dict[str, int], list[str], np.ndarray, int]:
    """ ЗАГРУЗКА ПРЕДОБУЧЕННОЙ МОДЕЛИ W2V/SG/NS"""
    # Надпись
    cf.print_inscription('ЗАГРУЗКА ПРЕДОБУЧЕННОЙ МОДЕЛИ W2V/SG/NS')
    # Проверка и загрузка модели W2V/SkipGram из gensim
    pretrained_w2v_model = nn.W2V_SkipGram_utils.check_and_load_model(PRETRAINED_MODELS_DIR_PATH)
    # Получение данных из модели
    pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size = nn.W2V_SkipGram_utils.get_vocab_and_embeddings(pretrained_w2v_model)
    # Разделительная черта
    cf.print_sub_line()
    
    return pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size

def get_train_val_test_data(clean_requests: list[list[str]], val_size: float = 0.2, test_size: float = 0.1):
    """ РАЗБИЕНИЕ КОРПУСА """
    # Надпись
    cf.print_inscription('РАЗБИЕНИЕ КОРПУСА НА ОБУЧАЮЩУЮ, ВАЛИДАЦИОННУЮ И ТЕСТОВУЮ ВЫБОРКИ')
    # Процесс
    try:
        train_data, val_data, test_data = nn.split_dataset(clean_requests, val_size=val_size, test_size=test_size)
    except Exception as e:
        cf.print_critical_error('Невозможно сформировать датасет.', prefix='\n')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)
    # Разделительная черта
    cf.print_sub_line()
    
    return train_data, val_data, test_data

def continue_learning_model() -> bool:
    # Продолжать обучение или нет
    while True:
        """ ДИАЛОГОВОЕ ОКНО ДЛЯ ВЫБОРА ФОРМАТА ОБУЧЕНИЯ МОДЕЛИ"""
        # Надпись
        cf.print_inscription('ДИАЛОГОВОЕ ОКНО ДЛЯ ВЫБОРА ФОРМАТА ОБУЧЕНИЯ МОДЕЛИ')
        cf.print_menu_option('[Разработчик]: Меню выбора модели -> Выбор формата обучения')
        choice = input('Вы хотите продолжить обучение модели?\n[Y - да, продолжить; N - нет, обучить с нуля]: ').strip().lower()
        if choice == 'y':
            cf.print_success('Вы выбрали продолжить обучение модели на основе сохраненного файла чекпоинта.')
            cf.print_sub_line()
            return True
        elif choice == 'n':
            cf.print_success('Вы выбрали обучить новую модель с нуля.')
            cf.print_sub_line()
            return False
        else:
            cf.print_error('Пожалуйста, введите Y или N.')
            cf.print_sub_line()
  
def normalized_probabilities_by_raw_counts(anomaly_probs: dict[str, int]) -> dict[str, int]:
    # Перевод количества обращений в вероятности
    total = sum(anomaly_probs.values()) # Абсолютное количиество обращений
    for k in anomaly_probs:
        anomaly_probs[k] = int(round((anomaly_probs[k] / total) * 100)) if total > 0 else 0
    # Нормализация "сырых" значений
    total = sum(anomaly_probs.values()) # Абсолютные вероятности для обращений (сумма != 100)
    if total > 0:
        normalized = {k: (v / total) * 100 for k, v in anomaly_probs.items()}
    else:
        normalized = {k: 0 for k in anomaly_probs}
    # Округление "сырых" значений вниз
    rounded = {k: int(v) for k, v in normalized.items()}
    # Компенсация остатков наибольшим элементам
    diff = 100 - sum(rounded.values())
    if diff > 0:
        fractional_parts = {k: normalized[k] - rounded[k] for k in rounded}
        for k in sorted(fractional_parts, key=fractional_parts.get, reverse=True)[:diff]:
            rounded[k] += 1
    anomaly_probs = rounded

    return anomaly_probs

""" ДОПОЛНИТЕЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ """
def prepare_training_session_for_model(TRAINED_MODEL_DIR_PATHS: list[str]) -> Optional[tuple[str, bool]]:
    for path in TRAINED_MODEL_DIR_PATHS:
        if not os.path.isdir(path):
            cf.print_critical_error('Папка не найдена.', prefix='\n')
            cf.print_critical_error(f'Путь: {path}', end='\n\n')
            exit(1)
    
    # Обучение/продолжение обучения модели
    CONTINUE_LEARNING = False
    # Ввод названия папки для текущей модели
    while True:
        """ ПОДГОТОВКА К ОБУЧЕНИЮ МОДЕЛИ """
        # Надпись
        cf.print_inscription('ПОДГОТОВКА К ОБУЧЕНИЮ МОДЕЛИ')
        cf.print_menu_option('[Разработчик]: Меню выбора модели -> Выбор папки')
        MODEL_DIR = input('Введите название папки для сохранения данных текущей модели (регистр учитывается только при создании папки)\n[~exit~ - выход]: ')
        if MODEL_DIR == "~exit~":
            cf.print_success('Выход без выбора папки.')
            cf.print_sub_line()
            return None
        else:
            confirm = input(f'Вы уверены, что хотите выбрать папку {MODEL_DIR}?\n[YES - да, [Any|Enter] - нет]: ')
            if confirm != 'YES':
                cf.print_success('Вы решили выбрать другую папку.')
                cf.print_sub_line()
                continue
        if cf.is_valid_folder_name(MODEL_DIR): # Корректное название
            cf.print_success('Вы выбрали папку для сохранения данных новой модели.')
            cf.print_success('Название папки:', end=' '); cf.print_key_info(f'{MODEL_DIR}')
            cf.print_sub_line()
            
            full_paths = [os.path.join(path, MODEL_DIR) for path in TRAINED_MODEL_DIR_PATHS]
            if any(os.path.isdir(p) and os.listdir(p) for p in full_paths):
                cf.print_warn('Папка с указанным названием уже существует для сохранения данных модели. Она не пустая.')
                cf.print_sub_line()
                if os.path.isdir(full_paths[0]) and cf.has_pt_files(full_paths[0]):
                    CONTINUE_LEARNING = continue_learning_model()
                if not CONTINUE_LEARNING:
                    cf.print_error('Во избежание перемешивания данных обучаемых моделей введите другое название папки. Спасибо!')
                    cf.print_sub_line()
                    continue
        
            """ ВАЖНАЯ ИНФОРМАЦИЯ """
            # Надпись
            cf.print_inscription('ВАЖНАЯ ИНФОРМАЦИЯ')
                
            TRAINED_MODEL_DIR_PATHS = full_paths
        
            for path in TRAINED_MODEL_DIR_PATHS:
                if not os.path.isdir(path):
                    os.makedirs(path)
                    cf.print_info(f'Папка "{path}" была создана.')
                    
            cf.print_info('Чекпоинты и визуализация модели после обучения будут сохранены в папках:')
            cf.print_info('Чекпоинты:', end=' '); cf.print_key_info(f'{TRAINED_MODEL_DIR_PATHS[0]}')
            cf.print_info('Визуализация:', end=' '); cf.print_key_info(f'{TRAINED_MODEL_DIR_PATHS[1]}')
                     
            # Разделительная черта
            cf.print_sub_line()        

            break
        else:
            cf.print_error('Название папки содержит недопустимые символы или их комбинацию. Пожалуйста, попробуйте снова.')
            cf.print_sub_line()
    
    return TRAINED_MODEL_DIR_PATHS, CONTINUE_LEARNING

def prepare_init_model_w2v_sg_ns(train_data: list[list[str]], pretrained_word2idx: dict[str, int], pretrained_idx2word: list[str], 
                                 pretrained_embeddings: np.ndarray, vector_size: int) -> tuple[dict[str, int], list[str], torch.Tensor, torch.device, torch.Tensor]:
    """ ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ МОДЕЛИ W2V/SG/NS """
    # Надпись
    cf.print_inscription('ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ МОДЕЛИ W2V/SG/NS')
    
    # Построение словаря и матрицы эмбеддингов
    try:
        word2idx, idx2word, embedding_matrix = nn.build_vocab_and_embeddings(train_data, pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size)
    except Exception as e:
        cf.print_critical_error('Невозможно построить словарь и матрицу эмбеддингов.', prefix='\n')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)
    # Устройство CPU/GPU для вычислений
    device = nn.get_device()
    # Распределение вероятностей для выбора негативных примеров
    noise_dist = nn.W2V_SkipGram_NS.get_noise_distribution(train_data, word2idx, device=device)
    
    # Разделительная черта
    cf.print_sub_line()

    return word2idx, idx2word, embedding_matrix, device, noise_dist

def prepare_build_model_w2v_sg_ns(word2idx: dict[str, int], vector_size: int, device: torch.device, embedding_matrix: torch.Tensor, noise_dist: torch.Tensor) -> nn.W2V_SkipGram_NS.SkipGramNS:
    """ СОЗДАНИЕ МОДЕЛИ W2V/SG/NS"""
    # Надпись
    cf.print_inscription('СОЗДАНИЕ МОДЕЛИ W2V/SG/NS')
    
    # Создание модели
    model = nn.W2V_SkipGram_NS.SkipGramNS(len(word2idx), vector_size, device=device, embedding_weights=embedding_matrix, noise_dist=noise_dist).to(device)
    
    cf.print_info('Модель Word2Vec -> SkipGram with Negative Sampling создана:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            cf.print_info('Параметры: ', end=''); cf.print_key_info(f'{name}', end='; ')
            print('размер: ', end=''); cf.print_key_info(f'{param.size()}')
    
    # Разделительная черта
    cf.print_sub_line()

    return model

def choose_file_from_directory(dir_path: str, ext: str='.pt', mode: str='dev') -> Optional[str]:
    if not os.path.isdir(dir_path):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь: {dir_path}', end='\n\n')
        exit(1)

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(f'{ext}') and not f.startswith('~$')]
    
    if not files:
        cf.print_warn(f'В папке нет {ext} файлов.')
        cf.print_warn(f'Путь: {dir_path}')
        cf.print_sub_line()
        return None

    while True:
        """ ВЫБОР ФАЙЛА В ПАПКЕ """
        # Надпись
        cf.print_inscription('ВЫБОР ЧЕКПОИНТА МОДЕЛИ') if mode=='dev' else  cf.print_inscription('ВЫБОР ФАЙЛА ДАННЫХ')
        
        # Процесс
        cf.print_info('Найдены файлы в папке.')
        cf.print_info(f'Путь: {dir_path}')
        cf.print_menu_option('[Разработчик]: Меню выбора модели -> Выбор чекпоинта') if mode=='dev' else cf.print_menu_option('[Пользователь]: Выбор формата входных данных -> Выбор файла данных')
        print('Выберите файл:')
        for idx, filename in enumerate(files, 1):
            print(f'{idx}. {filename}')
        print('0. Выход')
        try:
            choice = int(input('Введите номер файла для выбора [0 - выход]: '))
            if choice == 0:
                cf.print_success('Выход без выбора файла в меню выше.')
                cf.print_sub_line()
                return None
            elif 1 <= choice <= len(files):
                chosen_file_path = os.path.join(dir_path, files[choice - 1])
                cf.print_success('Вы выбрали файл.')
                cf.print_success('Путь к файлу:', end=' '); cf.print_key_info(f'{chosen_file_path}')
                cf.print_info('Анализируемая колонка в файле:', end=' '); cf.print_key_info(f'{config.file_column}')
                cf.print_sub_line()
                return chosen_file_path
            else:
                cf.print_error(f'Пожалуйста, введите число от 0 до {len(files)}.')
                cf.print_sub_line()
        except ValueError:
            cf.print_error('Некорректный ввод. Введите число.')
            cf.print_sub_line()

def choose_folder_from_directory(base_path: str) -> Optional[str]:
    if not os.path.isdir(base_path):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь: {base_path}', end='\n\n')
        exit(1)

    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    if not folders:
        cf.print_warn('В директори нет папок.')
        cf.print_warn(f'Путь: {base_path}')
        cf.print_sub_line()
        return None

    while True:
        """ ПАПКА МОДЕЛИ С ЧЕКПОИНТАМИ """
        # Надпись
        cf.print_inscription('ВЫБОР ПАПКИ МОДЕЛИ С ЧЕКПОИНТАМИ')
        cf.print_menu_option('[Разработчик]: Меню выбор модели -> Выбор папки')
        # Процесс
        print("Выберите папку:")
        for i, folder in enumerate(folders, 1):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path) and not os.listdir(folder_path):
                print(f"{i}. {folder} [пустая]")
            else:
                print(f"{i}. {folder}")
        print("0. Выход")
        try:
            choice = int(input("Введите номер папки для выбора [0 - выход]: "))
            if choice == 0:
                cf.print_success('Выход без выбора папки.')
                cf.print_sub_line()
                return None
            elif 1 <= choice <= len(folders):
                chosen_folder_path = os.path.join(base_path, folders[choice - 1])
                cf.print_success(f'Вы выбрали папку: {chosen_folder_path}')
                cf.print_sub_line()
                return chosen_folder_path
            else:
                cf.print_error(f'Пожалуйста, введите число от 0 до {len(folders)}.')
                cf.print_sub_line()
        except ValueError:
            cf.print_error('Некорректный ввод. Введите число.')
            cf.print_sub_line()

def load_checkpoint_device_model_info_w2v_sg_ns(checkpoint_file_path: str) -> tuple[dict, torch.device]:
    if not os.path.isfile(checkpoint_file_path):
        cf.print_critical_error('Файл не найден.', prefix='\n')
        cf.print_critical_error(f'Путь к файлу: {checkpoint_file_path}', end='\n\n')
        exit(1)
    
    # Устройство CPU/GPU
    device = nn.get_device()
    # Чекпоинт
    cf.print_info('Чекпонит модели загружается. Пожалуйста, подождите...')
    checkpoint = torch.load(checkpoint_file_path, map_location=device, weights_only=False)
    cf.print_info('Чекпоинт модели был загружен!', end='\n\n')
    
    # Вывод информации о модели
    cf.print_key_info('Информация о подгруженной модели W2V/SG/NS')
    
    # Дата и время
    cf.print_info('Дата и время:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['datetime']}")
    
    # Папки с чекпоинтами и графиками
    cf.print_info('Папка с чекпоинтами:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['paths']['checkpoints']}")
    
    cf.print_info('Папка с графиками:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['paths']['graphs']}")
    
    cf.print_info('Текущий чекпоинт:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['paths']['checkpoint_file']}")
    
    # Параметры модели
    cf.print_info('Параметры модели:')
    for name, shape in checkpoint['info']['model_parameters'].items():
        cf.print_info(f"   Параметр: {name}; размер:", end=' ')
        cf.print_key_info(f'{shape}')
    
    # Размер словаря
    cf.print_info('Размер словаря:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['vocab_size']}")
    
    # Размер матрицы эмбеддингов
    cf.print_info('Размер матрицы эмбеддингов:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['embedding']['matrix_shape']}")
    
    # Размер эмбеддинга
    cf.print_info('Размер эмбеддинга:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['embedding']['size']}")
    
    # Информация о тренировке
    cf.print_info('Информация о тренировке:')
    cf.print_info(f"   Устройство:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['device']}")
    cf.print_info(f"   Формат обучения:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['format']}")
    cf.print_info(f"   Использование графиков:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['graph']}")
    cf.print_info(f"   Количество негативных примеров:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['samples']}")
    cf.print_info(f"   Размер окна:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['window_size']}")
    cf.print_info(f"   Размер батча:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['batch_size']}")
    cf.print_info(f"   Количество эпох:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['epochs']}")
    cf.print_info(f"   Распределение вероятностей выбора негативных примеров:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['noise_dist']}")
    cf.print_info(f"   Размер тренировочного датасета:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['len_dataset_train']}")
    cf.print_info(f"   Размер валидационного датасета:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['training']['len_dataset_val']}")
    
    # Информация о тестировании
    cf.print_info('Информация о тестировании:')
    cf.print_info(f"   Размер тестирующего датасета:", end=' ')
    cf.print_key_info(f"{checkpoint['info']['testing']['len_dataset_test']}")

    # Примечание
    cf.print_info('Примечание:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['note']}", end='\n\n')
    
    return checkpoint, device

def load_checkpoint_device_model_info_kohonen(checkpoint_file_path: str) -> tuple[dict, torch.device]:
    if not os.path.isfile(checkpoint_file_path):
        cf.print_critical_error('Файл не найден.', prefix='\n')
        cf.print_critical_error(f'Путь к файлу: {checkpoint_file_path}', end='\n\n')
        exit(1)
        
    # Устройство CPU/GPU
    device = nn.get_device()
    # Чекпоинт
    cf.print_info('Чекпонит модели загружается. Пожалуйста, подождите...')
    checkpoint = torch.load(checkpoint_file_path, map_location=device, weights_only=False)
    cf.print_info('Чекпоинт модели был загружен!', end='\n\n')
    
    # Вывод информации о модели
    cf.print_key_info('Информация о подгруженной модели Кохонена')
    
    # Информация
    cf.print_info('Сведения:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['about']}")

    # Дата и время
    cf.print_info('Дата и время:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['datetime']}")
    
    # Сохраненная эпоха
    cf.print_info('Сохраненная эпоха:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['saved_epoch'] + 1}")
    
    # Устройство
    cf.print_info('Устройство:', end=' ')
    cf.print_key_info(f"{checkpoint['info']['device']}")
    
    # Папки с чекпоинтами и графиками
    cf.print_info('Папка с чекпоинтами:', end=' ')
    cf.print_key_info(f"{checkpoint['paths']['checkpoints']}")
    
    cf.print_info('Папка с графиками:', end=' ')
    cf.print_key_info(f"{checkpoint['paths']['map']}")
    
    cf.print_info('Текущий чекпоинт:', end=' ')
    cf.print_key_info(f"{checkpoint['paths']['checkpoint_file']}")
    
    # Параметры модели
    cf.print_info('Параметры модели:')
    cf.print_info('   Количество строк решетки:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['x_size']}")
    cf.print_info('   Количество столбцов решетки:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['y_size']}")
    cf.print_info('   Размерность пространства:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['input_dim']}")
    cf.print_info('   Скорость обучения:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['lr']}")
    cf.print_info('   Радиус влияния на соседей:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['sigma']}")
    cf.print_info('   Размерность матрицы весов:', end=' ')
    cf.print_key_info(f"{tuple(checkpoint['model_parameters']['weights'].shape)}")
    cf.print_info('   Размерность матрицы индексации нейронов:', end=' ')
    cf.print_key_info(f"{tuple(checkpoint['model_parameters']['neuron_locations'].shape)}")
    cf.print_info('   Количество эпох:', end=' ')
    cf.print_key_info(f"{checkpoint['model_parameters']['num_epochs']}")
    cf.print_info('   Строить SOM-карту:', end=' ')
    cf.print_key_info(f"{'Да' if checkpoint['model_parameters']['map_flag'] else 'Нет'}")
    
    # Информация об обучении
    cf.print_info('Информация об обучении:')
    cf.print_info('   Формат обучения:', end=' ')
    cf.print_key_info(f"{checkpoint['training']['format']}", end='\n\n')
    
    return checkpoint, device

def prepare_save_clusterization_settings(save_dir_path: str) -> Optional[str]:
    if not os.path.isdir(save_dir_path):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь: {save_dir_path}', end='\n\n')
        exit(1)
    
    # Ввод названия папки для сохранения настроек
    while True:
        """ ПОДГОТОВКА К СОХРАНЕНИЮ НАСТРОЕК КЛАСТЕРИЗАЦИИ """
        # Надпись
        cf.print_inscription('ПОДГОТОВКА К СОХРАНЕНИЮ НАСТРОЕК КЛАСТЕРИЗАЦИИ')
        cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена -> Выбор папки для сохранения настроек кластеризации')
        cf.print_warn('Если вы выйдите без выбора папки, то придется заново называть кластеры.')
        CLUSTER_DIR = input('Введите название папки для сохранения настроек кластеризации (регистр учитывается только при создании папки)\n[~exit~ - выход]: ')
        if CLUSTER_DIR == "~exit~":
            cf.print_success('Выход без выбора папки.')
            cf.print_sub_line()
            return None
        else:
            confirm = input(f'Вы уверены, что хотите выбрать папку {CLUSTER_DIR}?\n[YES - да, [Any|Enter] - нет]: ')
            if confirm != 'YES':
                cf.print_success('Вы решили выбрать другую папку.')
                cf.print_sub_line()
                continue
        if cf.is_valid_folder_name(CLUSTER_DIR): # Корректное название
            cf.print_success('Вы выбрали папку для сохранения настроек кластеризации.')
            cf.print_success('Название папки:', end=' '); cf.print_key_info(f'{CLUSTER_DIR}')
            
            full_path = os.path.join(save_dir_path, CLUSTER_DIR)
            if os.path.isdir(full_path) and os.listdir(full_path):
                cf.print_error('Во избежание перемешивания данных настроек кластеризации введите другое название папки. Спасибо!')
                cf.print_sub_line()
                continue
            save_dir_path = full_path
            os.makedirs(save_dir_path)
            cf.print_info('Настройки кластеризации будут сохранены в папке.')
            cf.print_info('Папка:', end=' '); cf.print_key_info(f'{save_dir_path}')
            # Разделительная черта
            cf.print_sub_line()        

            break
        else:
            cf.print_error('Название папки содержит недопустимые символы или их комбинацию. Пожалуйста, попробуйте снова.')
            cf.print_sub_line()
    
    return save_dir_path


""" ОСНОВНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ """
def train_w2v_sg_ns_model(train_data: list[list[str]], val_data: list[list[str]], test_data: list[list[str]], word2idx: dict[str, int], idx2word: list[str],
                          TRAINED_MODEL_DIR_PATHS: str, model: nn.W2V_SkipGram_NS.SkipGramNS, device: torch.Tensor) -> None:
    """ ОБУЧЕНИЕ/ДООБУЧЕНИЕ МОДЕЛИ W2V/SG/NS """
    # Надпись
    cf.print_inscription('ОБУЧЕНИЕ/ДООБУЧЕНИЕ МОДЕЛИ W2V/SG/NS')
    # Генерация пар для формирования датасета
    cf.print_key_info('>Пары обучающей выборки<')
    pairs_train = nn.generate_skipgram_data_pairs(train_data, word2idx, window_size=config.window_size_w2v_sg_ns)
    cf.print_key_info('>Пары валидационной выборки<')
    pairs_val = nn.generate_skipgram_data_pairs(val_data, word2idx, window_size=config.window_size_w2v_sg_ns)
    cf.print_key_info('>Пары тестирующей выборки<')
    pairs_test = nn.generate_skipgram_data_pairs(test_data, word2idx, window_size=config.window_size_w2v_sg_ns)
    # Формирование датасета
    dataset_train = nn.W2V_SkipGram_NS.SkipGramDataset(pairs_train, device=device)
    dataset_val = nn.W2V_SkipGram_NS.SkipGramDataset(pairs_val, device=device)
    dataset_test = nn.W2V_SkipGram_NS.SkipGramDataset(pairs_test, device=device)
    
    cf.print_key_info('>Датасет выборок<')
    cf.print_info('Обучающая:\t', end=''); cf.print_key_info(f'{len(dataset_train)}')
    cf.print_info('Валидационная:\t', end=''); cf.print_key_info(f'{len(dataset_val)}')
    cf.print_info('Тестирующая:\t', end=''); cf.print_key_info(f'{len(dataset_test)}')
    
    if len(dataset_train) == 0 or len(dataset_val) == 0 or len(dataset_test) == 0:
        cf.print_critical_error('Слишком мало данных! Словарный запас не позволяет корректно обработать датасет для модели.', end='\n\n', prefix='\n')
        exit(1)
        
    cf.print_warn('Некоторые гиперпараметры и настройки модели берутся из файла конфига ./core_functions/config.py.', end='\n\n', prefix='\n')

    # Дообучение модели
    nn.W2V_SkipGram_NS.train_skipgram_ns(TRAINED_MODEL_DIR_PATHS, word2idx, idx2word, model, dataset_train, dataset_val, dataset_test, device=device)
    # Разделительной черта
    cf.print_sub_line()

def continue_train_w2v_sg_ns_model(checkpoint_file_path: str) -> None:
    if checkpoint_file_path:
        """ ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ W2V/SG/NS """
        # Надпись
        cf.print_inscription('ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ W2V/SG/NS')
        
        # Информация о модели
        checkpoint, device = load_checkpoint_device_model_info_w2v_sg_ns(checkpoint_file_path)

        # Запуск обучения модели
        nn.W2V_SkipGram_NS.continue_train_skipgram_ns(checkpoint, device=device)
        
        # Вывод разделительной черты
        cf.print_sub_line()
    else:
        return None

def get_saved_w2v_sg_ns_model(model_dir_path: str, pretrained_word2idx: dict[str, int], pretrained_idx2word: list[str], pretrained_embeddings: np.ndarray, 
                              vector_size: int, clean_requests: list[list[str]]) -> Optional[tuple[nn.W2V_SkipGram_NS.SkipGramNS, dict[str, int], list[str], 
                                                                                                   int, nn.W2V_SkipGram_NS.SkipGramDataset, torch.device]]:
    # Выбор папки с сохраненной моделью
    while True:
        chosen_model_path = choose_folder_from_directory(model_dir_path)
        if not chosen_model_path:
            break
        
        chosen_model_file_path = choose_file_from_directory(chosen_model_path)
        if chosen_model_file_path:
            """ ПОЛУЧЕНИЕ МОДЕЛИ W2V/SG/NS ИЗ ЧЕКПОИНТА"""
            # Надпись
            cf.print_inscription('ПОЛУЧЕНИЕ МОДЕЛИ W2V/SG/NS ИЗ ЧЕКПОИНТА')
            
            # Чекпоинт и устройство
            checkpoint, device = load_checkpoint_device_model_info_w2v_sg_ns(chosen_model_file_path)
            
            # Создание модели
            cf.print_info('Создание модели W2V/SG/NS. Пожалуйста, подождите...')
            model = nn.W2V_SkipGram_NS.SkipGramNS(checkpoint['model_params']['vocab_size'], checkpoint['model_params']['embedding_dim'],
                                                       device=device, embedding_weights=None, noise_dist=checkpoint['model_params']['noise_dist']).to(device)
            model.load_state_dict(checkpoint['state_model'])
            cf.print_info('Модель была создана!')
            
            # Словарь модели
            word2idx = cf.load_from_json(checkpoint['vocab']['word2idx_path'])
            idx2word = cf.load_from_json(checkpoint['vocab']['idx2word_path'])
            # Количество негативных примеров
            n_samples = checkpoint['model_params']['n_samples']
            # Тестирующий датасет
            dataset_test = cf.load_from_pickle(checkpoint['dataloader']['test_dataset_path'])

            # Разделительная строка
            cf.print_sub_line()
            
            return model, word2idx, idx2word, n_samples, dataset_test, device
        continue
    
    # Выбор загрузки модели по умолчанию
    while True:
        """ МОДЕЛЬ W2V/SG/NS ПО УМОЛЧАНИЮ """
        # Надпись
        cf.print_inscription('МОДЕЛЬ W2V/SG/NS ПО УМОЛЧАНИЮ')
        cf.print_menu_option('[Разработчик]: Выбор модели W2V/SG/NS -> Модель по умолчанию')
        choice = input("Вы хотите загрузить модель W2V/SG/NS по умолчанию?\n[Y - да; N - нет, выход в меню для выбора модели]: ").strip().lower()
        if choice == 'y':
            cf.print_success('Вы выбрали загрузить модель W2V/SG/NS по умолчанию.')
            cf.print_sub_line()
            
            # Инициализация модели
            word2idx, idx2word, embedding_matrix, device, noise_dist = prepare_init_model_w2v_sg_ns([], pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size)
            # Создание модели
            default_model = prepare_build_model_w2v_sg_ns(word2idx, vector_size, device=device, embedding_matrix=embedding_matrix, noise_dist=noise_dist)
            # Разбиение выборки на обучающую, валидационную и тестовую
            _, _, test_data = get_train_val_test_data(clean_requests, val_size=0, test_size=0) # [], [], clean_requests
            
            """ ИНФОРМАЦИЯ О ТЕСТИРОВАНИИ """
            # Надпись
            cf.print_inscription('ИНФОРМАЦИЯ О ТЕСТИРОВАНИИ')
            # Генерация пар тестирующей выборки
            cf.print_key_info('>Пары тестирующей выборки<')
            pairs_test = nn.generate_skipgram_data_pairs(test_data, word2idx, window_size=config.window_size_w2v_sg_ns)
            # Формирование тестирующего датасета
            dataset_test = nn.W2V_SkipGram_NS.SkipGramDataset(pairs_test, device=device)
            # Выводы
            cf.print_info('Датасет для тестирующей выборки создан.')
            cf.print_info('Размер тестирующей выборки:', end=' '); cf.print_key_info(f'{len(dataset_test)}')
            # Проверка датасета
            if len(dataset_test) == 0:
                cf.print_critical_error('Слишком мало данных! Словарный запас не позволяет корректно обработать датасет для тестирования модели.', end='\n\n', prefix='\n')
                exit(1)
            
            cf.print_warn('Количество негативных примеров берется из файла конфига ./core_nn/config.py.', prefix='\n')
            
            # Разделительная черта
            cf.print_sub_line()

            return default_model, word2idx, idx2word, config.pretrained_n_samples_w2v_sg_ns, dataset_test, device
        elif choice == 'n':
            cf.print_success('Вы выбрали выйти без выбора модели.')
            cf.print_sub_line()
            return None
        else:
            cf.print_error("Пожалуйста, введите Y (да) или N (нет).")
            cf.print_sub_line()

def test_w2v_sg_ns_model(model: nn.W2V_SkipGram_NS.SkipGramNS, dataset_test: nn.W2V_SkipGram_NS.Dataset, n_samples: int, device: torch.device) -> None:
    """ ТЕСТИРОВАНИЕ МОДЕЛИ W2V/SG/NS """
    # Надпись
    cf.print_inscription('ТЕСТИРОВАНИЕ МОДЕЛИ W2V/SG/NS')
    
    if len(dataset_test) == 0:
        cf.print_critical_error('Модель обучена не достаточно! Словарный запас не позволяет корректно обработать датасет для тестирования модели.', end='\n\n', prefix='\n')
        exit(1)
    
    cf.print_warn('Размер батча для тестирования берется из файла конфига ./core_nn/config.py.', end='\n\n')

    # Тестирование модели
    nn.W2V_SkipGram_NS.test_skipgram_ns(model, dataset_test, n_samples, device=device)
    # Разделительная черта
    cf.print_sub_line()
    
def prepare_trained_embeddings(model: nn.W2V_SkipGram_NS.SkipGramNS) -> np.ndarray:
    """ ОБНОВЛЕНИЕ И ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ """
    # Надпись
    cf.print_inscription('ОБНОВЛЕНИЕ И ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ')
    # Обновление данных
    trained_embeddings = nn.get_trained_embeddings(model)
    # Вывод разделительной черты
    cf.print_sub_line()
    
    return trained_embeddings

def find_similar_words(word2idx: dict[str, int], idx2word: list[str], embeddings: np.ndarray) -> None:
    while True:
        """ ПОХОЖИЕ СЛОВА """
        # Надпись
        cf.print_inscription('ПОХОЖИЕ СЛОВА')
        cf.print_menu_option('[Разработчик]: Выбор модели W2V/SG/NS -> Поиск похожих слов')
        # Процесс
        cf.print_info("Введите слово в формате 'слово_POS', где POS — часть речи.")
        cf.print_info("Допустимые части речи: NOUN (существительное), ADJ (прилагательное), VERB (глагол),")
        cf.print_info("PART (частица), ADP (предлог), ADV (наречие), SCONJ (подчинительный союз),")
        cf.print_info("DET (определитель), CCONJ (сочинительный союз).")
        word = input('Введите слово [~exit~ - выход]: ')
        if word == '~exit~':
            cf.print_success('Вы выбрали завершить проверку похожих слов.')
            break
        if word not in word2idx:
            cf.print_warn('Слово отсутствует в модели.')
            cf.print_warn(f'Слово: {word}')
            cf.print_sub_line()
            continue
        
        idx = word2idx[word]
        word_vector = embeddings[idx].reshape(1, -1)                    # (1, embedding_dim)
        similarities = cosine_similarity(word_vector, embeddings)[0]    # (1, N)
        similar_indices = similarities.argsort()[::-1]                  # Сортировка по убыванию
        similar_indices = [i for i in similar_indices if i != idx][:10] # Индексы похожих слов и значений схожести

        cf.print_info(f"Топ 10 слов, похожих на '{word}':", prefix='\n')
        for i in similar_indices:
            cf.print_info(f"{idx2word[i]}:", end=' '); cf.print_key_info(f"{similarities[i]:.4f}")
        cf.print_sub_line()
        
def get_saved_kohonen_model(model_dir_path: str) -> Optional[tuple[nn.SOM, list[list[str]]]]:
    # Выбор папки с сохраненной моделью
    while True:
        chosen_model_path = choose_folder_from_directory(model_dir_path)
        if not chosen_model_path:
            break
        
        chosen_model_file_path = choose_file_from_directory(chosen_model_path)
        if chosen_model_file_path:
            """ ПОЛУЧЕНИЕ МОДЕЛИ КОХОНЕНА ИЗ ЧЕКПОИНТА"""
            # Надпись
            cf.print_inscription('ПОЛУЧЕНИЕ МОДЕЛИ КОХОНЕНА ИЗ ЧЕКПОИНТА')
            
            # Чекпоинт и устройство
            checkpoint, device = load_checkpoint_device_model_info_kohonen(chosen_model_file_path)
            
            # Создание модели
            cf.print_info('Создание модели Кохонена. Пожалуйста, подождите...')
            model = nn.SOM()
            model.load_som_by_checkpoint(checkpoint, device=device)
            cf.print_info('Модель была создана!')
           
            # Тренировочный датасет
            dataset_train = cf.load_from_pickle(checkpoint['paths']['data_path'])

            # Разделительная строка
            cf.print_sub_line()
            
            return model, dataset_train, checkpoint['paths']['map']
    return None


""" ДЕЛЕГИРОВАННЫЕ ФУНКЦИИ """
### РАЗРАБОТЧИК ###
def dev_handle_mode_train_w2v_sg_ns_model(clean_requests: list[list[str]], TRAINED_MODEL_DIR_PATHS: list[str],
                                          pretrained_word2idx: dict[str, int], pretrained_idx2word: list[str], pretrained_embeddings: np.ndarray,
                                          vector_size: int) -> None:
    ### Этап II. Подготовка к обучению модели.
    result = prepare_training_session_for_model(TRAINED_MODEL_DIR_PATHS)
    if result is not None:
        TRAINED_MODEL_DIR_PATHS, CONTINUE_LEARNING = result
    else:
        return None
        
    ### Этап III. Разбиение выборки на обучающую, валидиационную и тестовую.
    train_data, val_data, test_data = get_train_val_test_data(clean_requests, val_size=0.2, test_size=0.1)
        
    ### ЭТАП IV. Инициализация, создание и обучение[дообучение]/продолжение обучения модели.
    if not CONTINUE_LEARNING: # Обучение с начала
        word2idx, idx2word, embedding_matrix, device, noise_dist = prepare_init_model_w2v_sg_ns(train_data, pretrained_word2idx, pretrained_idx2word,
                                                                                                pretrained_embeddings, vector_size)
        # Создание модели
        model = prepare_build_model_w2v_sg_ns(word2idx, vector_size, device=device, embedding_matrix=embedding_matrix, noise_dist=noise_dist)
        # Обучение модели
        train_w2v_sg_ns_model(train_data, val_data, test_data, word2idx, idx2word, TRAINED_MODEL_DIR_PATHS, model, device=device)
    else:
        # Выбор чекпоинта модели
        checkpoint_file_path = choose_file_from_directory(TRAINED_MODEL_DIR_PATHS[0])
        # Продолжение обучения модели
        continue_train_w2v_sg_ns_model(checkpoint_file_path)

def dev_handle_mode_get_and_test_w2v_sg_ns_model(clean_requests: list[list[str]], TRAINED_MODEL_DIR_PATHS: list[str], 
                                                 pretrained_word2idx: dict[str, int], pretrained_idx2word: list[str], pretrained_embeddings: np.ndarray,
                                                 vector_size: int) -> Optional[tuple[dict[str, int], list[str], np.ndarray]]:
    ### Этап II. Создание модели на основе сохраненной/обученной модели.
    result = get_saved_w2v_sg_ns_model(TRAINED_MODEL_DIR_PATHS[0], pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size, clean_requests)
    if result is None:
        return None
    model, word2idx, idx2word, n_samples, dataset_test, device = result
        
    ### Этап III. Тестирование модели
    while True:
        """ ДИАЛОГОВОЕ ОКНО ВЫБОРА ТЕСТИРОВАНИЯ МОДЕЛИ """
        # Надпись
        cf.print_inscription('ДИАЛОГОВОЕ ОКНО ВЫБОРА ТЕСТИРОВАНИЯ МОДЕЛИ')
        cf.print_menu_option('[Разработчик]: Мкню выбора модели W2V/SG/NS -> Тест модели')
        # Ввод
        cf.print_warn('Тестирование модели может занимать продолжительное время.')
        cf.print_warn('Выход из тестирования модели является аварийным.')
        choice = input('Вы хотите протестировать модель?\n[Y - да, N - нет]: ').strip().lower()
        # Выбор
        if choice == 'y':
            cf.print_success('Вы выбрали протестировать модель.')
            cf.print_sub_line()
            test_w2v_sg_ns_model(model, dataset_test, n_samples, device=device)
            break
        elif choice == 'n':
            cf.print_success('Вы выбрали не тестировать модель.')
            cf.print_sub_line()
            break
        else:
            cf.print_error('Пожалуйста, введите Y или N.')
            cf.print_sub_line()
            
    ### Этап IV. Получение обученных эмбеддингов
    trained_embeddings = prepare_trained_embeddings(model)
            
    return word2idx, idx2word, trained_embeddings

def dev_handle_mode_find_similar_words(word2idx: dict[str, int], idx2word: list[str], trained_embeddings: np.ndarray) -> None:
    ### Этап V. Проверка схожести слов в моделе
    cf.print_inscription('ПРОВЕРКА СХОЖЕСТИ СЛОВ')
    cf.print_menu_option('[Разработчик]: Меню выбора модели W2V/SG/NS -> Проверка схожести слов')
    check_similar_words = input('Вы хотите протестировать модель на близкие слова?\n[YES - да, <Any|Enter> - нет]: ')
    if check_similar_words == 'YES':
        cf.print_success('Вы выбрали произвести проверку схожести слов модели.')
        cf.print_sub_line()
        find_similar_words(word2idx, idx2word, trained_embeddings)
    else:
        cf.print_success('Вы выбрали не производить проверку схожести слов модели.')
    cf.print_sub_line()

def dev_handle_mode_get_request_embeddings(corpus: list[list[str]], word2idx: dict[str, int], 
                                           embeddings: np.ndarray, vector_size: int) -> Optional[tuple[dict[str, float], np.ndarray]]:
    """ ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ ОБРАЩЕНИЙ """
    # Вывод надписи
    cf.print_inscription('ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ ОБРАЩЕНИЙ')
    ### Этап VI. Получение эмбеддингов обращений
    cf.print_menu_option('[Разработчик]: Меню выбора модели W2V/SG/NS -> Эмбеддинги обращений')
    mode = input('Сформировать эмбеддинги обращений?\n[NO - нет, <Any|Enter> - да]: ')
    if mode != 'NO':
        tfidf_dict = nn.compute_tf_idf(corpus, word2idx)
        request_embeddings = nn.get_request_embeddings(corpus, word2idx, embeddings, vector_size, tfidf_dict)
        cf.print_main_line()
        return tfidf_dict, request_embeddings
    else:
        cf.print_sub_line()
        return None

def dev_handle_mode_train_kohonen_model(request_embeddings: np.ndarray, TRAINED_MODEL_DIR_PATHS: list[str]):
    ### Этап VII. Подготовка к обучению модели
    result = prepare_training_session_for_model(TRAINED_MODEL_DIR_PATHS)
    if result is not None:
        TRAINED_MODEL_DIR_PATHS, CONTINUE_LEARNING = result
    else:
        return None
    
    ### ЭТАП VIII. Инициализация, создание и обучение/продолжение обучения модели.
    if not CONTINUE_LEARNING: # Обучение с начала
        """ ОБУЧЕНИЕ МОДЕЛИ КОХОНЕНА """
        # Надпись
        cf.print_inscription('ОБУЧЕНИЕ МОДЕЛИ КОХОНЕНА')
        # Устройство CPU/GPU
        device = nn.get_device()
        # Создание модели
        model = nn.SOM(config.x_size_kohonen, config.y_size_kohonen, request_embeddings.shape[1], map_flag=True, device=device,
                       sigma=None, learning_rate=config.learning_rate_kohonen, num_epochs=config.epochs_kohonen)
        # Тренировка модели
        model.train(request_embeddings, TRAINED_MODEL_DIR_PATHS, count_save_epoch=10)
        # Разделительная черта
        cf.print_sub_line()
    else: # Продолжение обучения
        # Выбор чекпоинта модели
        checkpoint_file_path = choose_file_from_directory(TRAINED_MODEL_DIR_PATHS[0])
        # Продолжение обучения модели
        if checkpoint_file_path:
            """ ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ КОХОНЕНА """
            # Надпись
            cf.print_inscription('ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ КОХОНЕНА')
            # Информация о модели
            checkpoint, device = load_checkpoint_device_model_info_kohonen(checkpoint_file_path)
            # Создание модели
            model = nn.SOM()
            # Запуск обучения модели
            model.continue_train(checkpoint, device=device, count_save_epoch=10)
            # Вывод разделительной черты
            cf.print_sub_line()
        else:
            return None
    
def dev_handle_mode_get_cluster_and_visualize_kohonen_model(TRAINED_MODEL_DIR_PATHS: list[str]) -> Optional[tuple[nn.SOM, list[tuple[int, int]]]]:
    ### Этап IX. Создание модели на основе сохраненной модели.
    result = get_saved_kohonen_model(TRAINED_MODEL_DIR_PATHS[0])
    if result is None:
        return None
    model, request_embeddings, save_dir_map_path = result
        
    ### Этап X. Кластеризация обращений и визуализация
    """ КЛАСТЕРИЗАЦИЯ ОБРАЩЕНИЙ И ВИЗУАЛИЗАЦИЯ"""
    # Надпись
    cf.print_inscription('КЛАСТЕРИЗАЦИЯ ОБРАЩЕНИЙ И ВИЗУАЛИЗАЦИЯ')
    # Кластеризация
    cf.print_info('Производится кластеризация обращений. Пожалуйста, подождите...')
    cluster_assignments = model.assign_clusters(request_embeddings)
    cf.print_info('Кластеризация обращений завершена!', end='\n\n')
    # Визуализация
    new_map_flag = False
    if not model.map_flag:
        cf.print_info('Модель содержит настройку не визуализировать кластеры.')
        cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена -> Визуализация обращений')
        mode = input('Вы уверены, что не хотите построить SOM-карту?\n[YES - да, <Any|Enter> - визуализировать]: ')
        if mode == 'YES':
            cf.print_success('Вы выбрали не визуализировать модель Кохонена.')
            cf.print_sub_line()
            return model, cluster_assignments
        else:
            new_map_flag = True
            cf.print_success('Вы выбрали визуализировать модель Кохонена.', end='\n\n')
    cf.print_info('Производится визуализация обращений. Пожалуйста, подождите...')
    model.visualize_clusters(save_dir_map_path, cluster_assignments, new_map_flag)
    cf.print_info('Визуализация обращений завершена!')
    # Разделительная черта
    cf.print_sub_line()
    
    return model, cluster_assignments

def dev_handle_mode_setting_up_clusters_and_save(model: nn.SOM, requests: list[str], clean_requests: list[list[str]], 
                                                 request_embeddings: np.ndarray, tfidf_dict: dict[str, float],
                                                 cluster_assignments: list[tuple[int, int]],
                                                 top_n: int = 30, top_anomaly_orig_n: int = 10) -> Optional[tuple[dict[tuple[int,int],np.ndarray], dict[tuple[int,int], dict[str, int]]]]:
    ### ЭТАП XI. Настройки кластеризации и сохранение
    # Переменные
    processed_clusters = 0
    x_size = model.x_size
    y_size = model.y_size
    # Словари для хранения эмбеддингов кластеров и их названий
    cluster_embeddings = {}
    cluster_names = {}
    # Количество непустых кластеров
    total_clusters = sum(1 for x in range(x_size) for y in range(y_size) if any(cluster_x == x and cluster_y == y for cluster_x, cluster_y in cluster_assignments))
    
    # Перебираем все кластеры решетки
    for x in range(x_size):
        for y in range(y_size):
            # Все обращения, попавшие в текущий кластер
            cluster_orig_requests = []
            cluster_requests = []
            cluster_indices = []
            for idx, (cluster_x, cluster_y) in enumerate(cluster_assignments):
                if cluster_x == x and cluster_y == y:
                    cluster_orig_requests.append(requests[idx])
                    cluster_requests.append(clean_requests[idx])
                    cluster_indices.append(idx)
            # Если кластер пустой, пропускаем
            if not cluster_requests and (x != 0 or y != 0):
                continue
            # Подсчитываем важность слов по tf-idf в этом кластере
            word_tfidf = Counter()
            for req in cluster_requests:
                for word in req:
                    # Если слово есть в tf-idf словаре, увеличиваем счетчик
                    if word in tfidf_dict:
                        word_tfidf[word] += tfidf_dict[word]
            # Выбираем топ N самых значимых слов
            top_words_all = [word for word, _ in word_tfidf.most_common(top_n)]

            # Обработка зарезервированных кластеров
            if (x,y) == (0,0):
                # Надпись
                cf.print_inscription('НАСТРОЙКА КЛАСТЕРОВ')
                cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена -> Настройка кластеров')
                # Оставшиеся кластеры
                remaining_clusters = total_clusters - processed_clusters
                cf.print_info('Осталось настроить кластеров:', end=' '); cf.print_key_info(f'{remaining_clusters}', end='\n\n')
                # Номер кластера и количество обращений
                cf.print_info('Кластер:', end=' '); cf.print_key_info(f'({x}, {y})')
                cf.print_info('Он содержит обращений:', end=' '); cf.print_key_info(f'{len(cluster_requests)}')
                # Название кластера
                cluster_name = {}
                cluster_name["Аномалия: Неопознанное обращение"] = 100
                cf.print_success('Автоматически назначено название кластеру:')
                for key, value in cluster_name.items():
                    cf.print_success(f'  {key:<30} — {value:>3}%')
                cluster_names[(x,y)] = cluster_name
                # Эмбеддинг кластера
                cluster_embeddings[(x, y)] = np.zeros(request_embeddings.shape[1])
                processed_clusters += 1
                # Разделительная черта
                if remaining_clusters != 1: cf.print_sub_line()
                continue

            # Обработка кластеров с автоназваниями
            anomaly_probs = {description: 0 for _, description in config.target_words_for_cluster_kohonen.items()} # Вероятности для стандартных классов (высокое, низкое, скачки) + аномалия
            cluster_anomaly_requests = []      # Аномальные обращения кластера
            cluster_anomaly_orig_requests = [] # Исходные аномальные обращения кластера
            for idx in range(len(cluster_requests)):
                current_cluster_request = cluster_requests[idx]
                matched = False
                for target_words_str, description in config.target_words_for_cluster_kohonen.items(): # Стандартные классы обращений
                    target_words = target_words_str.split(',')
                    if any(target_word in current_cluster_request for target_word in target_words) and (len(cluster_orig_requests[idx]) > config.default_request_len):
                        if not matched: matched = True
                        anomaly_probs[description] += 1
                if not matched:
                    cluster_anomaly_orig_requests.append(cluster_orig_requests[idx])
                    cluster_anomaly_requests.append(cluster_requests[idx])
            # Если не стандартных обращений нет, то автоматическое название кластера
            if len(cluster_anomaly_requests) == 0:
                # Нормализация вероятностей
                anomaly_probs = normalized_probabilities_by_raw_counts(anomaly_probs)
                # Надпись
                cf.print_inscription('НАСТРОЙКА КЛАСТЕРОВ')
                cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена -> Настройка кластеров')
                # Оставшиеся кластеры
                remaining_clusters = total_clusters - processed_clusters
                cf.print_info('Осталось настроить кластеров:', end=' '); cf.print_key_info(f'{remaining_clusters}', end='\n\n')
                # Номер кластера и количество обращений
                cf.print_info('Кластер:', end=' '); cf.print_key_info(f'({x}, {y})')
                cf.print_info('Он содержит обращений:', end=' '); cf.print_key_info(f'{len(cluster_requests)}')
                # Название кластера
                cluster_name = {k: int(v) for k, v in anomaly_probs.items()}
                cf.print_success('Автоматически назначено название кластеру:')
                for key, value in cluster_name.items():
                    cf.print_success(f'  {key:<30} — {value:>3}%')
                cluster_names[(x,y)] = cluster_name
                # Эмбеддинг кластера
                cluster_embeddings[(x, y)] = np.mean(request_embeddings[cluster_indices], axis=0)
                processed_clusters += 1
                # Разделительная черта
                if remaining_clusters != 1: cf.print_sub_line()
                continue

            # Иначе добавляется характеристика для нестандртного обращения
            # Обработка остальных кластеров
            top_current = (len(top_words_all) if top_n <= len(top_words_all) else len(top_words_all))
            while True:
                """ НАСТРОЙКА КЛАСТЕРОВ """
                # Надпись
                cf.print_inscription('НАСТРОЙКА КЛАСТЕРОВ')
                cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена -> Настройка кластеров')
                # Оставшиеся кластеры
                remaining_clusters = total_clusters - processed_clusters
                cf.print_info('Осталось настроить кластеров:', end=' '); cf.print_key_info(f'{remaining_clusters}', end='\n\n')
                # Выводим значимые слова и запрашиваем название кластера
                cf.print_info('Кластер:', end=' '); cf.print_key_info(f'({x}, {y})')
                cf.print_info('Он содержит обращений:', end=' '); cf.print_key_info(f'{len(cluster_requests)}')
                if len(top_words_all) != 0:
                    cf.print_key_info(f'>Топ {top_current} значимых слов<')
                    cf.print_info('Слова:')
                    for word in top_words_all:
                        cf.print_info(f'   {word}')
                    print()
                else:
                    cf.print_key_info('Кластер не содержит значимых слов!', end='\n\n')

                # Отображение исходных обращений
                top_anomaly_orig_current = (top_anomaly_orig_n if top_anomaly_orig_n <= len(cluster_anomaly_requests) else len(cluster_anomaly_requests))
                print_orig_requests = input(f'Отобразить топ-{top_anomaly_orig_current} аномальных исходных обращений?\n[YES - да, ~exit~ - выйти без сохранения, <Any|Enter> - нет]: ')
                if print_orig_requests == "YES":
                    cf.print_success(f'Вы выбрали отобразить топ-{top_anomaly_orig_current} аномальных исходных обращений.', prefix='\n')
                    for num_req in range(top_anomaly_orig_current):
                        cf.print_info(f'[{num_req + 1}]: {cluster_anomaly_orig_requests[num_req]}')
                        cf.print_info(f'[{num_req + 1}]:', end=' '); cf.print_key_info(f'{cluster_anomaly_requests[num_req]}')
                    print()
                elif print_orig_requests == "~exit~":
                    cf.print_success('Вы выбрали выйти без сохранения из настройки кластеров.')
                    cf.print_sub_line()
                    return None
                else:
                    cf.print_success(f'Вы выбрали не отображать топ-{top_anomaly_orig_current} оригинальных обращений.', end='\n\n')

                cluster_user_name = input(f"Введите название для аномалии этого кластера [~exit~ - выйти без сохранения, <Enter> - {config.default_anomaly_name}]: ")
                if not cluster_user_name:
                    cluster_user_name = config.default_anomaly_name
                if cluster_user_name == '~exit~':
                    cf.print_success('Вы выбрали выйти без сохранения из настройки кластеров.')
                    cf.print_sub_line()
                    return None
                mode = input("Вы уверены, что хотите дать такое название аномалии кластера?\n[NO - нет, <Enter> - да, ~exit~ - выйти без сохранения]: ")
                if not mode:
                    # Нестандартное обращение
                    anomaly_probs[cluster_user_name] = len(cluster_anomaly_requests)
                    # Нормализация вероятностей
                    anomaly_probs = normalized_probabilities_by_raw_counts(anomaly_probs)
                    # Информация
                    cf.print_success('Вы выбрали дать название аномалии кластера:', end=' '); cf.print_key_info(f'{cluster_user_name}')
                    # Название кластера
                    cluster_name = {k: int(v) for k, v in anomaly_probs.items()}
                    cf.print_success('Назначено название кластеру:')
                    for key, value in cluster_name.items():
                        cf.print_success(f'  {key:<60} — {value:>3}%')
                    cluster_names[(x,y)] = cluster_name
                    # Эмбеддинг кластера
                    cluster_embeddings[(x, y)] = np.mean(request_embeddings[cluster_indices], axis=0)
                    processed_clusters += 1
                    # Разделительная черта
                    if remaining_clusters != 1: cf.print_sub_line()
                    break
                elif mode == '~exit~':
                    cf.print_success('Вы выбрали выйти без сохранения из настройки кластеров.')
                    cf.print_sub_line()
                    return None
                elif mode == 'NO':
                    cf.print_success('Вы выбрали изменить название аномалии кластера.')
                    cf.print_sub_line()
                    continue
                else:
                    cf.print_error('Некорректный ввод. Пожалуйста, попробуйте снова')
                    cf.print_sub_line()
                    continue
                    
    cf.print_success('ВСЕ КЛАСТЕРЫ БЫЛИ НАСТРОЕНЫ!', prefix='\n')
    cf.print_sub_line()
    
    return cluster_embeddings, cluster_names

def dev_handle_mode_save_clusterization_settings(cluster_embeddings: dict[tuple[int, int], np.ndarray], cluster_names: dict[tuple[int, int], dict[str, int]], save_dir_path: str,
                                                 tfidf_dict: dict[str, float], word2idx: dict[str, int], idx2word: list[str], trained_embeddings):
    result = prepare_save_clusterization_settings(save_dir_path)
    if result is None:
        return None
    save_dir_path = result
    
    """ СОХРАНЕНИЕ НАСТРОЕК КЛАСТЕРИЗАЦИИ """
    # Надпись
    cf.print_inscription('СОХРАНЕНИЕ НАСТРОЕК КЛАСТЕРИЗАЦИИ')
    # Процесс
    cf.print_info('Выполняется сохранение настроек кластеризации. Пожалуйста, подождите...')
    save_cluster_embeddings_path = os.path.join(save_dir_path, config.cluster_embeddings_file)
    save_cluster_names_path = os.path.join(save_dir_path, config.cluster_names_file)
    cf.save_to_json(cf.convert_keys_to_str(cluster_embeddings), save_cluster_embeddings_path)
    cf.save_to_json(cf.convert_keys_to_str(cluster_names), save_cluster_names_path)
    cf.print_info('Сохранение настроек кластеризации выполнено!')
    cf.print_info('Пути к файлам:')
    cf.print_info('   Эмбеддинги кластеров:', end=' '); cf.print_key_info(f'{save_cluster_embeddings_path}')
    cf.print_info('   Названия кластеров:', end=' '); cf.print_key_info(f'{save_cluster_names_path}', end='\n\n')
    # Сохранение прочих настроек
    cf.print_info('Выполняется сохранение остальных настроек кластеризации. Пожалуйста, подождите...')
    save_word2idx_path = os.path.join(save_dir_path, config.word2idx_file)
    save_idx2word_path = os.path.join(save_dir_path, config.idx2word_file)
    save_tfidf_dict_path = os.path.join(save_dir_path, config.tfidf_dict_file)
    save_embeddings_path = os.path.join(save_dir_path, config.embeddings_file)
    cf.save_to_json(word2idx, save_word2idx_path)
    cf.save_to_json(idx2word, save_idx2word_path)
    cf.save_to_json(tfidf_dict, save_tfidf_dict_path)
    cf.save_to_pickle(trained_embeddings, save_embeddings_path)
    cf.print_info('Сохранение остальных настроек кластеризации выполнено!')
    cf.print_info('Пути к файлам:')
    cf.print_info('   Словарь word2idx:', end=' '); cf.print_key_info(f'{save_word2idx_path}')
    cf.print_info('   Словарь idx2word:', end=' '); cf.print_key_info(f'{save_idx2word_path}')
    cf.print_info('   Словарь tfidf:', end=' '); cf.print_key_info(f'{save_tfidf_dict_path}')
    cf.print_info('   Эмбеддинги:', end=' '); cf.print_key_info(f'{save_embeddings_path}')
    cf.print_warn('Укажите папку, которая будет использоваться пользователем, в файле config.py.')
    cf.print_warn(f'Текущая настроенная папка:',end=' '); cf.print_key_info(f'{save_dir_path}')
    # Разделительная черта
    cf.print_sub_line()

### ПОЛЬЗОВАТЕЛЬ ###
def user_handle_mode_get_clusterization_settings(dir_path: str) -> tuple[dict[str, int], list[str], dict[str, float], np.ndarray, dict[tuple[int,int], np.ndarray], dict[tuple[int,int], dict[str,int]]]:
    if not os.path.isdir(dir_path):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь: {dir_path}', end='\n\n')
        exit(1)
    
    """" ЗАГРУЗКА НАСТРОЕК КЛАСТЕРИЗАЦИИ """
    # Надпись
    cf.print_inscription('ЗАГРУЗКА НАСТРОЕК КЛАСТЕРИЗАЦИИ')
    # Процесс
    cf.print_info('Загружаются данные настроек кластеризации. Пожалуйста, подождите...')
    word2idx = cf.load_from_json(os.path.join(dir_path, config.word2idx_file))
    idx2word = cf.load_from_json(os.path.join(dir_path, config.idx2word_file))
    tfidf_dict = cf.load_from_json(os.path.join(dir_path, config.tfidf_dict_file))
    embeddings = cf.load_from_pickle(os.path.join(dir_path, config.embeddings_file))
    cluster_embeddings = cf.convert_keys_to_tuple(cf.load_from_json(os.path.join(dir_path, config.cluster_embeddings_file)))
    cluster_names = cf.convert_keys_to_tuple(cf.load_from_json(os.path.join(dir_path, config.cluster_names_file)))
    cf.print_info('Данные загружены!')
    cf.print_sub_line()

    return word2idx, idx2word, tfidf_dict, embeddings, cluster_embeddings, cluster_names

def user_handle_mode_get_udpipe_model(PRETRAINED_MODELS_DIR_PATH: str) -> Model:
    """ МОДЕЛЬ UDPIPE"""
    # Надпись
    cf.print_inscription('МОДЕЛЬ UDPIPE')
    # Процесс
    model = load_udpipe_model(PRETRAINED_MODELS_DIR_PATH)
    # Разделительная черта
    cf.print_main_line()
    
    return model

def user_handle_mode_get_request_embeddings(data: list[list[str]], word2idx: dict[str, int], embeddings: np.ndarray, tfidf_dict: dict[str, float]) -> np.ndarray:
    """ ПОЛУЧЕНИЕ ЭМБЕДДИНГА(-ОВ) ОБРАЩЕНИЯ(-Й) """
    # Вывод надписи
    cf.print_inscription('ПОЛУЧЕНИЕ ЭМБЕДДИНГА(-ОВ) ОБРАЩЕНИЯ(-Й)')
    # Процесс
    request_embeddings = nn.get_request_embeddings(data, word2idx, embeddings, embeddings.shape[1], tfidf_dict)
    # Разделительная черта
    cf.print_sub_line()
    return request_embeddings

def user_handle_mode_get_clusters_for_embeddings(request_embeddings: np.ndarray, cluster_embeddings: dict[tuple[int, int], np.ndarray],
                                                 cluster_names: dict[tuple[int, int], dict[str, int]]) -> list[dict[str, int]]:
    # Cловарь эмбеддингов кластеров -> список координат -> массив эмбеддингов кластеров
    cluster_coords = list(coord for coord in cluster_embeddings if coord != (0, 0))
    cluster_vecs = np.array([cluster_embeddings[coord] for coord in cluster_coords])  # shape: (n_clusters - 1, vector_dim)

    results = []
    for emb in request_embeddings:
        if np.all(emb == 0): # Неопозанное обращение
            nearest_coord = (0,0)
        else:
            # Вычисление Евклидового расстояния до каждого кластера
            distances = np.linalg.norm(cluster_vecs - emb, axis=1) # shape: (n_clusters - 1,)
            nearest_index = np.argmin(distances)                   # int
            nearest_coord = cluster_coords[nearest_index]          # shape: (1,1)
        
        cluster_name = cluster_names.get(nearest_coord, "Неизвестный кластер")
        results.append(cluster_name)

    return results

def user_handle_save_clustered_results(data_file_path: str, save_dir_path: str, request_clusters: list[dict[str,int]], target_column: str) -> None:
    """ СОХРАНЕНИЕ ОБРАБОТАННОГО ФАЙЛА """
    # Надпись
    cf.print_inscription('СОХРАНЕНИЕ ОБРАБОТАННОГО ФАЙЛА')
    # Процесс
    cf.print_info('Осуществляется сохранение обработанного файла. Пожалуйста, подождите...')
    new_data_file_path = cf.read_file_add_column_and_save(data_file_path, save_dir_path, request_clusters, target_column)
    if new_data_file_path is None:
        return
    cf.print_info('Сохранение обработанного файла завершено!')
    cf.print_info('Путь к файлу:', end=' '); cf.print_key_info(f'{new_data_file_path}')
    # Разделительная строка
    cf.print_sub_line()

def user_handle_preprocessed_single_request(request: str, model: Model) -> list[str]:
    """ ПРЕДОБРАБОТКА ОБРАЩЕНИЯ """
    # Надпись
    cf.print_inscription('ПРЕДОБРАБОТКА ОБРАЩЕНИЯ')
    # Процесс
    clean_request = process_single_request(request, model)
    # Разделительная черта
    cf.print_sub_line()
    
    return clean_request

""" ОСНОВНЫЕ ФУНКЦИИ """
def dev_running_program() -> None:
    ### МЕНЕДЖЕРЫ ###
    cf.print_inscription('ФОРМИРОВАНИЕ ПАПОК')
    dir_manager = DirectoriesManager(config.structure, root_path='.')
    cf.print_sub_line()
    csv_manager = CSVManager(config.file_name, config.file_column)
    ### ПАПКИ ###
    SOURCE_DATA_DIR_PATH = dir_manager.get_dir_path('SourceData')
    SOURCE_DATA_CSV_DIR_PATH = dir_manager.get_dir_path('SourceDataCSV')
    PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH = dir_manager.get_dir_path('PreprocessedSourceDataCSV')
    PRETRAINED_MODELS_DIR_PATH = dir_manager.get_dir_path('PretrainedModels')
    TRAINED_MODEL_DIR_PATHS_W2V_SG_NS = [dir_manager.get_dir_path('SG_NS_Model_checkpoints'), dir_manager.get_dir_path('SG_NS_Model_graphs')]
    TRAINED_MODEL_DIR_PATHS_KOHONEN = [dir_manager.get_dir_path('Kohonen_Model_checkpoints'), dir_manager.get_dir_path('Kohonen_Model_map')]
    CLUSTERIZATION_SETTINGS_PATH = dir_manager.get_dir_path('ClusterizationSettings')
    
    ### НАЧАЛЬНЫЕ ДЕЙСТВИЯ ###
    ### Этап I. Получение обращений потребителей, предобработка, очистка данных и загрузка предобученной модели.
    requests = get_consumer_requests(csv_manager, SOURCE_DATA_DIR_PATH, SOURCE_DATA_CSV_DIR_PATH)
    preprocessed_requests = get_preprocessed_requests(csv_manager, requests, PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, PRETRAINED_MODELS_DIR_PATH)
    clean_requests = get_cleaned_requests_from_None(preprocessed_requests)
    pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size = load_and_get_pretrained_w2v_model(PRETRAINED_MODELS_DIR_PATH)

    """Выбор дальнейшей реализации """
    while True:
        default_word2idx, default_idx2word, default_embeddings, default_vector_size = pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size
        # Надпись
        cf.print_inscription('ДИАЛОГОВОЕ МЕНЮ ДЛЯ ВЫБОРА МОДЕЛИ ВЕКТОРИЗАЦИИ')
        cf.print_menu_option('[Разработчик]: Меню выбора модели W2V/SG/NS')
        # Переменные
        SHOULD_TRAIN = USE_PRETRAINED = False
        # Меню
        print('Выберите модель W2/SG/NS:')
        print('1 - Создание, обучение и сохранение модели W2V/SG/NS на основе предобученной модели gensim')
        print('2 - Создание, обучение и сохранение модели W2V/SG/NS с нуля')
        print('3 - Использование сохраненной модели W2V/SG/NS на основе чекпоинта/предобученной модели gensim')
        print('0 - Выход')
        cf.print_info('Обучение модели может занимать очень продолжительное время.')
        cf.print_info('В процессе обучения чекпоинты модели сохраняются в соответствующую папку.')
        cf.print_info('Прервать обучение можно только аварийным завершением программы.')
        cf.print_info('В дальнейшем можно продолжить обучение путем подгрузки чекпоинта модели на сохраненной эпохе.')
        mode = input('Ваш выбор: ')
        
        if mode == '1':
            cf.print_success('Вы выбрали создание, обучение и сохранение модели W2V/SG/NS на основе предобученной модели gensim.')
            cf.print_sub_line()
            SHOULD_TRAIN = True
            USE_PRETRAINED = True
        elif mode == '2':
            cf.print_success('Вы выбрали создание, обучение и сохранение модели W2V/SG/NS с нуля.')
            cf.print_sub_line()
            SHOULD_TRAIN = True
            USE_PRETRAINED = False
        elif mode == '3':
            cf.print_success('Вы выбрали использовать сохраненную готовую модель W2V/SG/NS на основе чекопинта/предобученной модели gensim.')
            cf.print_sub_line()
            SHOULD_TRAIN = False
            USE_PRETRAINED = False
        elif mode == '0':
            cf.print_success('Выход в главное меню.')
            return
        else:
            cf.print_error('Неверный ввод. Пожалуйста, попробуйте снова.')
            cf.print_sub_line()
            continue

        if SHOULD_TRAIN: # Обучение модели и возврат
            if not USE_PRETRAINED:
                try:
                    default_word2idx, default_idx2word, default_embeddings, default_vector_size = cf.get_empty_pretrained_data(vector_size=300)
                except Exception as e:
                    cf.print_critical_error('Невозможно обучить модель W2V/SG/NS с нуля.', prefix='\n')
                    cf.print_critical_error(f'Причина: {e}', end='\n\n')
                    exit(1)
            dev_handle_mode_train_w2v_sg_ns_model(clean_requests, TRAINED_MODEL_DIR_PATHS_W2V_SG_NS, default_word2idx, default_idx2word, default_embeddings, default_vector_size)
            continue
        # Выбор модели и получение данных
        result = dev_handle_mode_get_and_test_w2v_sg_ns_model(clean_requests, TRAINED_MODEL_DIR_PATHS_W2V_SG_NS, pretrained_word2idx, pretrained_idx2word, pretrained_embeddings, vector_size)
        if result is None:
            continue
        word2idx, idx2word, trained_embeddings = result
        
        # Проверка схожести слов
        dev_handle_mode_find_similar_words(word2idx, idx2word, trained_embeddings)
        
        # Формирование эмбеддингов обращений
        result = dev_handle_mode_get_request_embeddings(clean_requests, word2idx, trained_embeddings, vector_size)
        if result is None:
            continue
        tdidf_dict, request_embeddings = result
        
        """ Нейронная сеть Кохонена """
        while True:
            # Надпись 
            cf.print_inscription('ДИАЛОГОВОЕ МЕНЮ ДЛЯ ВЫБОРА МОДЕЛИ КЛАСТЕРИЗАЦИИ')
            cf.print_menu_option('[Разработчик]: Меню выбора модели Кохонена')
            # Меню
            print('Выберите модель Кохонена:')
            print('1 - Создание, обучение и сохранение модели Кохонена')
            print('2 - Использование сохраненной модели Кохонена на основе чекпоинта')
            print('0 - Выход')
            cf.print_info('Обучение модели может занимать очень продолжительное время.')
            cf.print_info('В процессе обучения чекпоинты модели сохраняются в соответствующую папку.')
            cf.print_info('Прервать обучение можно только аварийным завершением программы.')
            cf.print_info('В дальнейшем можно продолжить обучение путем подгрузки чекпоинта модели на сохраненной эпохе.')
            mode = input('Ваш выбор: ')
            
            if mode == '1':
                cf.print_success('Вы выбрали создание, обучение и сохранение модели Кохонена.')
                cf.print_sub_line()
                SHOULD_TRAIN = True
            elif mode == '2':
                cf.print_success('Вы выбрали использовать сохраненную готовую модель Кохонена на основе чекпоинта')
                cf.print_sub_line()
                SHOULD_TRAIN = False
            elif mode == '0':
                cf.print_success('Выход в меню выбора модели W2V/SG/NS')
                cf.print_main_line()
                break
            else:
                cf.print_error('Неверный ввод. Пожалуйста, попробуйте снова.')
                cf.print_sub_line()
                continue
            
            if SHOULD_TRAIN: # Обучение модели и возврат
                dev_handle_mode_train_kohonen_model(request_embeddings, TRAINED_MODEL_DIR_PATHS_KOHONEN)
                continue
            result = dev_handle_mode_get_cluster_and_visualize_kohonen_model(TRAINED_MODEL_DIR_PATHS_KOHONEN)
            if result is None:
                continue
            model, cluster_assignments = result
            
            # Настройка кластеров
            result = dev_handle_mode_setting_up_clusters_and_save(model, requests, clean_requests, request_embeddings, tdidf_dict, cluster_assignments)
            if result is None:
                continue
            cluster_embeddings, cluster_names = result
            
            # Сохранение настроенной кластеризации
            dev_handle_mode_save_clusterization_settings(cluster_embeddings, cluster_names, CLUSTERIZATION_SETTINGS_PATH, tdidf_dict, word2idx, idx2word, trained_embeddings)
            
def user_running_program() -> None:
    ### НАЧАЛЬНЫЕ ДЕЙСТВИЯ ###
    # Необходимые папки
    dir_manager = DirectoriesManager(config.structure, root_path='.', build=False)
    SOURCE_DATA_DIR_PATH = dir_manager.get_dir_path('SourceData')
    SOURCE_DATA_CSV_DIR_PATH = dir_manager.get_dir_path('SourceDataCSV')
    PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH = dir_manager.get_dir_path('PreprocessedSourceDataCSV')
    PRETRAINED_MODELS_DIR_PATH = dir_manager.get_dir_path('PretrainedModels')
    CLUSTERIZATION_SETTINGS_PATH = os.path.join(dir_manager.get_dir_path('ClusterizationSettings'), config.settings_dir_name)
    CLUSTERED_RESULTS_PATH = dir_manager.get_dir_path('ClusteredResults')
    # Получение настроек кластеризации
    word2idx, idx2word, tfidf_dict, embeddings, cluster_embeddings, cluster_names = user_handle_mode_get_clusterization_settings(CLUSTERIZATION_SETTINGS_PATH)
    # Модель UDPipe
    model = user_handle_mode_get_udpipe_model(PRETRAINED_MODELS_DIR_PATH)

    # Реализация
    while True:
        """ ДИАЛОГОВОЕ ОКНО ВЫБОРА ДАННЫХ """
        # Надпись
        cf.print_inscription('ДИАЛОГОВОЕ ОКНО ВЫБОРА ДАННЫХ')
        cf.print_menu_option('[Пользователь]: Выбор формата входных данных')
        # Меню
        print('Выберите формат входных данных:')
        print('1 - Файл таблицы Excel формата .xlsb')
        print('2 - Пользовательский ввод обращения')
        print('0 - Выход')
        mode = input('Ваш выбор: ')
        USER_INPUT = False
        if mode == '1':
            cf.print_success('Вы выбрали формат входных данных в виде .xlsb файла.')
            cf.print_sub_line()
            USER_INPUT = False
        elif mode == '2':
            cf.print_success('Вы выбрали формат входных данных в виде пользовательского ввода.')
            cf.print_sub_line()
            USER_INPUT = True
        elif mode == '0':
            cf.print_success('Выход в главное меню.')
            return
        else:
            cf.print_error('Неверный ввод. Пожалуйста, попробуйте снова.')
            cf.print_sub_line()
            continue
        
        # Из .xlsb файла
        if not USER_INPUT:
            # Выбор файла для анализа
            data_file_path = choose_file_from_directory(SOURCE_DATA_DIR_PATH, ext='.xlsb', mode='user')
            if data_file_path is None:
                continue
            # Менеджер .csv файлов
            csv_manager = CSVManager(os.path.basename(data_file_path), config.file_column)
            # Обработка
            requests = get_consumer_requests(csv_manager, SOURCE_DATA_DIR_PATH, SOURCE_DATA_CSV_DIR_PATH)
            preprocessed_requests = get_preprocessed_requests(csv_manager, requests, PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, PRETRAINED_MODELS_DIR_PATH, mode='user', load_model=False)
            clean_requests = get_cleaned_requests_from_None(preprocessed_requests)
            # Получение эмбеддингов обращений
            request_embeddings = user_handle_mode_get_request_embeddings(clean_requests, word2idx, embeddings, tfidf_dict)
            # Определение кластеров обращений
            request_clusters = user_handle_mode_get_clusters_for_embeddings(request_embeddings, cluster_embeddings, cluster_names)
            # Сохранение обработанного файла
            user_handle_save_clustered_results(data_file_path, CLUSTERED_RESULTS_PATH, request_clusters, config.file_column)
            continue
        while True:
            """ ПОЛЬЗОВАТЕЛЬСКИЙ ВВОД ОБРАЩЕНИЯ """
            # Надпись
            cf.print_inscription('ПОЛЬЗОВАТЕЛЬСКИЙ ВВОД ОБРАЩЕНИЯ')
            cf.print_menu_option('[Пользователь]: Выбор формата входных данных -> Пользовательский ввод')
            # Процесс
            request = input('Введите обращение:\n[~exit~ - выход]: ')
            if request == '~exit~':
                cf.print_success('Вы выбрали выйти из пользовательского ввода обращений.')
                cf.print_sub_line()
                break
            choice = input('Подтвердить ввод:\n[YES - да, ~exit~ - выход, <Any|Enter> - нет]: ')
            if choice == 'YES':
                cf.print_success('Вы выбрали проанализировать обращение.')
                cf.print_success('Обращение:', end=' '); cf.print_key_info(f'{request}')
                cf.print_sub_line()
                # Получение предобработанного обращения
                clean_request = user_handle_preprocessed_single_request(request, model)
                # Получение эмбеддинга обращения
                request_embedding = user_handle_mode_get_request_embeddings([clean_request], word2idx, embeddings, tfidf_dict)
                # Определение кластера обращения
                request_cluster = user_handle_mode_get_clusters_for_embeddings(request_embedding, cluster_embeddings, cluster_names)
                # Вывод кластера
                cf.print_inscription('КЛАСТЕР ОБРАЩЕНИЯ')
                cf.print_info('Данному обращению соответствует кластер:')
                for key, value in request_cluster[0].items():
                    cf.print_info(f'  {key:<30} — {value:>3}%')
                cf.print_main_line()
                continue
            elif choice=='~exit~':
                cf.print_success('Вы выбрали выйти из пользовательского ввода обращений.')
                cf.print_main_line()
                break
            else:
                cf.print_success('Вы выбрали ввести другое обращение.')
                cf.print_sub_line()
                continue