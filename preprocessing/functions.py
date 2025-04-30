import wget, os
from ufal.udpipe import Model, Pipeline
from collections import defaultdict
from datetime import datetime
import core_functions as cf
from core_functions import config
from core_functions.utils import CSVManager

# Вывод статистики
def print_stats(stats: dict, top_n: int = None):
    if top_n is None or top_n and len(stats.keys()) < top_n:
        cf.print_info("Статистика предобработки. Все", end=' '); cf.print_key_info(f"{len(stats.keys())}", end=' '); print("значений по убыванию:")
    else:
        cf.print_info("Статистика предобработки. Первые", end=' '); cf.print_key_info(f"{top_n}/{len(stats.keys())}", end=' '); print("значений по убыванию:")

    # Сортировка
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    # Вывод
    for i, (k, v) in enumerate(sorted_stats):
        if top_n and i >= top_n:
            break
        cf.print_info(f"{k:<30}: {v}")
    print()

# Организация статистики
def organize_stats(stats: dict) -> dict:
    # Сортировка
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    sorted_stats_dict = {key: value for key, value in sorted_stats}

    # Группировка по префиксу
    grouped_stats = defaultdict(dict)
    for key, value in sorted_stats_dict.items():
        if ':' in key:
            group = key.split(':')[0]
        elif '_' in key:
            group = key.split('_')[0]
        else:
            group = 'other'
        grouped_stats[group][key] = value

    return {
        "meta": {
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_entries": len(sorted_stats_dict),
        },
        "grouped_stats": grouped_stats
    }    

# Кастомные обработки
def custom_preprocess(token: str, lemma: str, pos: str, stats: dict) -> tuple[str, str]:
    changed = False
    token = token.lower()
    if lemma == "респодобный":
        print(token)
    if token.startswith('э/э'):
        lemma = 'электроэнергия'
        pos = 'NOUN'
        stats['custom:электроэнергия'] += 1
        changed = True
    if token in {'респ.', 'респ'}:
        lemma = 'республика'
        pos = 'NOUN'
        stats['custom:респ->республика'] += 1
        changed = True
    if token in {'тел.', 'тел'}:
        lemma = 'телефон'
        pos = 'NOUN'
        stats['custom:тел->телефон'] += 1
        changed = True
    if token == 'д':
        lemma = 'дом'
        pos = 'NOUN'
        stats['custom:д->дом'] += 1
        changed = True
    if token == 'c':
        lemma = 'село'
        pos = 'NOUN'
        stats['custom:с->село'] += 1
        changed = True
    if lemma == "телефон":
        pos = "NOUN"
        stats['custom:телефон->NOUN'] += 1
        changed = True
    if changed:
        stats['custom_total'] += 1
    return lemma, pos  

# Чистка токена
def clean_token(token: str, misc: str, stats: dict) -> str:
    out_token = token.strip().replace(' ', '')
    if token == "Файл" and 'SpaceAfter=No' in misc:
        stats['clean_token:skip_Файл'] += 1
        return None
    if len(token) <= 2 and not token.isdigit():
        stats['clean_token:skip_short'] += 1
        return None
    if out_token != token:
        stats['clean_token:changed'] += 1
    return out_token

#Чистка леммы
def clean_lemma(lemma: str, pos: str, stats: dict) -> str:
    original = lemma
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        stats['clean_lemma:skip_bad_format'] += 1
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith(('«', '»')):
            out_lemma = out_lemma[1:]
        if out_lemma.endswith(('«', '»', '!', '?', ',', '.')):
            out_lemma = out_lemma[:-1]
    if out_lemma != original:
        stats['clean_lemma:changed'] += 1
    return out_lemma

# Функция предобработки на основе ufal.udpipe.Pipeline
def process(pipeline: Pipeline, text = "Строка", keep_pos = True, keep_punct = False, stats: dict = None) -> list[str]:
    if stats is None:
        stats = defaultdict(int)

    entities = {'PROPN'}
    other_allowed_entities = {'NOUN', 'ADJ', 'VERB', 'PART', 'ADP', 'ADV', 'SCONJ', 'DET', 'CCONJ'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []
    
    # Обрабатываем текст и получаем результат в формате CONCLL-U
    try:
        processed = pipeline.process(text)
    except Exception as e:
        cf.print_critical_error("Ошибка при обработке обращения.", prefix='\n')
        cf.prunt_critical_error(f"Обращение: {text}")
        cf.print_critical_error(f"Исключение: {e}", end='\n\n')
        exit(1)

    # Пропускаем строки со служебной информацией из формата
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # Извлекаем из обработанного текста леммы, тэги (часть речи) и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if (len(t) != 10): # Если список короткий (не 10 - значение по умолчанию) - строчка не содержит нужного разбора, пропускаем
            stats['bad_lines'] += 1
            continue
        (_, token, lemma, pos, _, feats, _, _, _, misc) = t # Токен, лемма, часть речь, фичи и прочее
        lemma, pos = custom_preprocess(token, lemma, pos, stats) # Кастомная обработка
        token = clean_token(token, misc, stats) # Очистка токена
        lemma = clean_lemma(lemma, pos, stats) # Очистка леммы
        if not lemma or not token: # Если лемма или токен пустые, то пропускаем
            stats['empty_token_or_lemma'] += 1
            continue
        if pos in entities: # Если имя собственное - обрабатываем
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                stats['short_feats_PROPN'] += 1
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                stats['no_case_number_PROPN'] += 1
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN')
                    stats['compound_propn'] += 1
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                if past_lemma:
                    tagged_propn.append(past_lemma + '_PROPN')
                    stats['compound_propn'] += 1
                tagged_propn.append(f'{lemma}_{pos}')
        else: # Иначе
            if named: # Выгружаем из памяти
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN')   
                stats['compound_propn'] += 1
            # Заносим новое
            if pos in other_allowed_entities:
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                stats[f'unused_POS_{pos}'] += 1

    # Удаление имен собственных (они не нужны)
    tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PROPN']            

    if not keep_punct: # Не сохраняются знаки препинания
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos: # Не сохраняются тэги частей речи
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    
    stats['total_processed'] += 1

    return tagged_propn

# Загрузка модели UDPipe
def load_udpipe_model(PRETRAINED_MODELS_DIR_PATH: str) -> Model:
    if not os.path.isdir(PRETRAINED_MODELS_DIR_PATH):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь к папке: {PRETRAINED_MODELS_DIR_PATH}', end='\n\n')
        exit(1)
    
    cf.print_info('Проверяется наличие модели UDPipe для предобработки обращений потребителей.')
    cf.print_info('Модель:', end=' '); cf.print_key_info(f'{config.udpipe_model_filename}')
    cf.print_info('Пожалуйста, подождите...')
    
    # Путь к загружаемой предобученной UDPipe модели
    model_path = os.path.join(PRETRAINED_MODELS_DIR_PATH, config.udpipe_model_filename)
    
    # Проверка наличия модели
    if not os.path.isfile(model_path):
        udpipe_url_full =  f'{config.udpipe_url}{config.udpipe_model_filename}' #URL сайта для русскоязычных UDPipe моделей
        cf.print_warn(f'Модель {model_path} не обнаружена.')
        cf.print_info('Осуществляется попытка скачивания с сайта.'); 
        cf.print_info('Сайт:', end=' '); cf.print_key_info(f'{config.udpipe_url}')
        cf.print_info('Пожалуйста, подождите...', end='\n\n')
            
        # Скачивание модели из URL
        try:
            wget.download(udpipe_url_full, model_path)
            cf.print_info('Модель была успешно скачена:', end=' ', prefix='\n'); cf.print_key_info(config.udpipe_model_filename)
            cf.print_info('Путь к модели:', end=' '); cf.print_key_info(model_path, end='\n\n')
        except Exception as e:
            cf.print_critical_error(f'Невозможно скачать модель с сайта.')
            cf.print_critical_error(f'Cайт с моделью: {udpipe_url_full}')
            cf.print_critical_error(f'Причина: {e}', end='\n\n')
            exit(1)
    else:
        cf.print_info('Найдена модель:', end=' '); cf.print_key_info(config.udpipe_model_filename)
        cf.print_info('Путь к модели:', end=' '); cf.print_key_info(model_path, end='\n\n')

    # Загрузка модели        
    cf.print_info('Используется модель UDPipe для синтаксического и морфологического анализа обращений потребителей.')
    cf.print_info('Модель:', end=' '); cf.print_key_info(model_path)

    try:
        model = Model.load(model_path)
        return model
    except Exception as e:
        cf.print_critical_error('Невозможно загрузить модель.', prefix='\n')
        cf.print_critical_error(f'Модель: {model_path}')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)

# Предобработка обращений потребителей предобученной моделью UDPipe
def process_requests(requests: list[str], csv_manager: CSVManager, PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH: str, model: Model, mode: str) -> list[list[str]]:
    if not os.path.isdir(PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH):
        cf.print_critical_error('Папка не найдена.', prefix='\n')
        cf.print_critical_error(f'Путь к папке: {PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH}', end='\n\n')
        exit(1)

    cf.print_info('Предобработка обращений потребителей. Пожалуйста, подождите...')    
    # Создание пайплана
    try:
        process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu') # Конвейер обработки текста
    except Exception as e:
        cf.print_critical_error('Невозможно загрузить пайплан.', prefix='\n')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)
    # Пути для сохранения обработанных обращений для восстановления и статистики
    pkl_path = os.path.join(PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, f'{csv_manager.get_file_hash()}.pkl')
    stats_json_path = os.path.join(PREPROCESSED_SOURCE_DATA_CSV_DIR_PATH, f'{os.path.splitext(csv_manager.file_name)[0]}_{csv_manager.file_column}_stats.json')
    # Проверка ранее незавершенной обработки
    if os.path.isfile(pkl_path) and os.path.isfile(stats_json_path):
        preprocessed_requests = cf.load_from_pickle(pkl_path) # Ранее обработанные обращения
        stats_loaded = cf.load_from_json(stats_json_path)     # Ранее созданная статистика
        stats = defaultdict(int, {k: v for group in stats_loaded.get("grouped_stats", {}).values() for k, v in group.items()})
        cf.print_info(f"Загружено {len(preprocessed_requests)}/{len(requests)} ранее обработанных обращений.")
        cf.print_info(f'Обращения взяты из: {csv_manager.file_name} -> {csv_manager.file_column}')
        cf.print_info(f'Загружена сохраненная статистика по обработанным обращениям.')
    else:
        preprocessed_requests = []
        stats = defaultdict(int)
        if os.path.isfile(pkl_path) and not os.path.isfile(stats_json_path) or not os.path.isfile(pkl_path) and os.path.isfile(stats_json_path):
            cf.print_warn('Обработка обращений будет начата с начала. Отсутствуют ранее обработанные обращения или статистика!')

    # Обработка обращений
    unprocessed_requests = requests[len(preprocessed_requests):]
    for idx, text in enumerate(unprocessed_requests, start=len(preprocessed_requests)):
        # Обработка каждого обращения
        output = process(process_pipeline, text=text, keep_pos=True, keep_punct=False, stats=stats)
        preprocessed_requests.append(output)
        
        # Сохранение обработанных обращений и статистики
        cf.save_to_pickle(preprocessed_requests, pkl_path)
        cf.save_to_json(organize_stats(stats), stats_json_path)

        # Вывод процента обработки (каждые 100 обращений)
        if idx % 10 == 0:
            progress = (idx + 1) / len(requests) * 100
            cf.print_info(f"Предобработка обращений: {progress:.2f}% завершено...", end="", prefix='\r')

    cf.print_info("Предобработка обращений: 100% завершено!     ", end='\n\n', prefix='\r')

    cf.print_info('Полная статистика предобработки была сохранена в файл.')
    cf.print_info(f'Путь к файлу:', end=' '); cf.print_key_info(f'{stats_json_path}')
    cf.print_info('Вы можете посмотреть ее вручную.', end='\n\n')
    
    # Краткий вывод неорганизованной статистики
    cf.print_menu_option('[Разработчик]: Меню выбора модели W2V/SG/NS -> Отображение статистики') if mode=='dev' else cf.print_menu_option('[Пользователь]: Выбор формата входных данных [.xlsb] -> Отображение статистики')
    top_value = input('Вы хотите отобразить статистику?\n[<число> - отобразить первые N значений по убыванию, <Any> - нет]: ')
    try:
        top_value = int(top_value)
        cf.print_success('Вы выбрали отобразить вывод стастистики.', end='\n\n')
        print_stats(stats, top_n=top_value)
    except Exception:
        cf.print_success('Вы выбрали пропустить вывод стастистики.', end='\n\n')

    return preprocessed_requests

# Предобработка обращения потребителя предобученной моделью UDPipe
def process_single_request(request: str, model: Model) -> list[str]:
    cf.print_info('Предобработка обращения потребителя. Пожалуйста, подождите...')    
    # Создание пайплана
    try:
        process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu') # Конвейер обработки текста
    except Exception as e:
        cf.print_critical_error('Невозможно загрузить пайплан.', prefix='\n')
        cf.print_critical_error(f'Причина: {e}', end='\n\n')
        exit(1)
    # Обработка
    preprocessed_request = process(process_pipeline, text=request, keep_pos=True, keep_punct=False, stats=None)
    cf.print_info('Предобработка обращения потребителя завершена!')
    
    return preprocessed_request