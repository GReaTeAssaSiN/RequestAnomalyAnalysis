""" ИНИЦИАЛИЗАЦИЯ """
### ОБРАЩЕНИЯ/МОДЕЛЬ W2V/SG/NS ###
file_name = 'качество 2024.xlsb'
file_column = 'Суть обращения'
### ИЕРАРХИЯ ПАПОК ###
structure = {
    "DataFolder": {
        "SourceData": {},
        "ClusteredResults": {},
        "SourceDataCSV": {},
        "PreprocessedSourceDataCSV": {},
    },
    "ModelsFolder": {
        "PretrainedModels": {},
        "TrainedModels": {
            "SkipGram_NS": {
                "SG_NS_Model_checkpoints": {},
                "SG_NS_Model_graphs": {}
            },
            "Kohonen": {
                "Kohonen_Model_checkpoints": {},
                "Kohonen_Model_map": {}
            }
        }
    },
    "ClusterizationSettings": {}
}
### ДОПОЛНИТЕЛЬНО ###
suffix = '_tagged'

""" ПОДГРУЖАЕМЫЕ МОДЕЛИ """
### МОДЕЛЬ UDPIPE ###
udpipe_model_filename = 'udpipe_syntagrus.model'
udpipe_url = f'https://rusvectores.org/static/models/'

""" НЕЙРОННЫЕ СЕТИ """
### МОДЕЛЬ W2V/SkipGram/NS ###
w2v_model_filename = 'word2vec-ruscorpora-300'
# НАСТРОЙКИ
window_size_w2v_sg_ns = 10
n_samples_w2v_sg_ns = 5
batch_size_w2v_sg_ns = 512
epochs_w2v_sg_ns = 15
graph_w2v_sg_ns = True
# ФАЙЛЫ СОХРАНЕНИЯ 
dataset_train_file_name_w2v_sg_ns = 'dataset_train.pkl'
dataset_val_file_name_w2v_sg_ns = 'dataset_val.pkl'
dataset_test_file_name_w2v_sg_ns = 'dataset_test.pkl'
word2idx_file_name_w2v_sg_ns = 'word2idx.json'
idx2word_file_name_w2v_sg_ns = 'idx2word.json'
# ТЕСТИРОВАНИЕ + ПРЕДОБУЧЕННЫЕ НАСТРОЙКИ
test_batch_size_w2v_sg_ns = 8
pretrained_n_samples_w2v_sg_ns = 5

### МОДЕЛЬ КОХОНЕНА ###
# НАСТРОЙКИ
x_size_kohonen = 15
y_size_kohonen = 20
epochs_kohonen = 250
learning_rate_kohonen = 0.4
map_kohonen = True
# ФАЙЛЫ СОХРАНЕНИЯ
dataset_train_file_name_kohonen = 'dataset_train.pkl'
# АВТОМАТИЧЕСКОЕ НАЗВАНИЕ КЛАСТЕРОВ
default_request_len = 100
default_anomaly_name = 'Нестандартная формулировка обращения. Требует внимания'
target_words_for_cluster_kohonen = {
    "высокий_ADJ,/высокое_ADJ,повышаться_VERB,повышенно_ADV,повышенный_ADJ,повышенный_VERB": "Высокое напряжение в сети",
    "низкий_ADJ,/низкое_ADJ,низкоенапряжение_NOUN,низкоес_VERB,снижаться_VERB,падать_VERB,пониженно_ADV,пониженный_ADJ,пониженный_VERB,слабый_ADJ": "Низкое напряжение в сети",
    "скачка_NOUN,скачок_NOUN,скачки_NOUN,перепад_NOUN,перебой_DET,перебой_NOUN,перебой_VERB,скакать_VERB,мерцать_VERB,мерцание_NOUN,моргать_VERB,мигать_VERB,мигание_NOUN,нестабильный_ADJ": "Скачки напряжения в сети",
    "фаза_NOUN": "Проблемы с фазами"
}

""" ПАПКА С НАСТРОЙКАМИ """
settings_dir_name = 'BaseSettings'
### ФАЙЛЫ НАСТРОЕК
cluster_embeddings_file = 'cluster_embeddings.json'
cluster_names_file = 'cluster_names.json'
word2idx_file = 'word2idx.json'
idx2word_file = 'idx2word.json'
tfidf_dict_file = 'tfidf_dict.json'
embeddings_file = 'embeddings.pkl'
