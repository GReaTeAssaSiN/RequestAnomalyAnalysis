from typing import Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, datetime
import core_functions as cf
from core_functions import config
from .W2V_SkipGram_utils import print_loss_acc_info

# Распределение вероятностей выбора негативных примеров для Negative Sampling
def get_noise_distribution(train_data: list[list[str]], word2idx: dict[str, int], device: torch.device = 'cpu', power = 0.75) -> Optional[torch.Tensor]:
    cf.print_info('Производится формирование распределения вероятностей выбора негативных примеров')
    cf.print_info('Функция активации: Negative Sampling.')
    cf.print_info('Пожалуйста, подождите...')
    if not train_data:
        cf.print_warn('Обучающий корпус пустой. Будет использовано равномерное распределение по словам словаря для выбора негативных примеров!')
        return None
    
    # Получение всех слов словаря из обучающего корпуса (с повторениями)
    str_words = [word for request in train_data for word in request if word in word2idx]
    # Вычисление частот слов в корпусе и общей частоты для нормализации частот
    word_counts = Counter(str_words)
    total_count = sum(word_counts.values())
    
    # Массив нормализованных частот с учетом индекса слов
    word_freqs = np.zeros(len(word2idx), dtype=np.float32)
    for word, idx in word2idx.items():
        word_freqs[idx] = word_counts.get(word, 0) / total_count
    # Построение noise_distribution с искажением (частые слова - не так часто, редкие - не такие редкие)
    unigram_dist = word_freqs ** power
    noise_dist = unigram_dist/ unigram_dist.sum()

    cf.print_info('Формирование распределения вероятностей завершено!')
    # Приведение noise_dist к тензору
    return torch.tensor(noise_dist, dtype=torch.float32).to(device)

# Класс модели Word2Vec/SkipGram с функцией потерь Negative Sampling
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device: torch.device='cpu', embedding_weights: torch.Tensor=None, noise_dist=None):
        super(SkipGramNS, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.noise_dist = noise_dist
        self.device = device

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)

        if embedding_weights is not None:
            self.in_embeddings.weight.data.copy_(embedding_weights)
        else:
            self.in_embeddings.weight.data.uniform_(-1, 1)
        self.out_embeddings.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embeddings(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.out_embeddings(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ Генерация векторов шума размерности (batch_size, n_samples, embedding_dim) """
        if self.noise_dist is None:
            noise_dist = torch.ones(self.vocab_size, device=self.device) # Равномерное распределение
        else:
            noise_dist = self.noise_dist.to(self.device)                 # Кастомное распределение
        
        noise_words = torch.multinomial(noise_dist, batch_size * n_samples, replacement=True).to(self.device) # Получение случайных индексов негативных примеров
        
        # Вектора шума
        noise_vectors = self.out_embeddings(noise_words).view(batch_size, n_samples, self.embedding_dim)

        return noise_vectors
        
# Класс функции потерь Negative Sampling 
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors, device: torch.device = 'cpu'):
        # Перевод на нужное устройство
        input_vectors = input_vectors.to(device)
        output_vectors = output_vectors.to(device)
        noise_vectors = noise_vectors.to(device)
        # Приведение тензоров к нужной форме для выполнения операций
        batch_size, embed_size = input_vectors.shape
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        # Функция потерь для положительного примера
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log() # ln(sigm(...))
        out_loss = out_loss.squeeze()                                       # Приведение Тензора (batch_size, 1, 1) -> (batch_size,)
        # Функция потерь для негативных примеров
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log() # ln(sigm(-...))
        noise_loss = noise_loss.squeeze().sum(1)                                   # Приведение Тензора (batch_size, n_samples, 1) 
                                                                                   # -> (batch_size, n_samples)
                                                                                   # -> (batch_size,)
        # Результат потери (усреднение по батчу)
        return -(out_loss + noise_loss).mean()

# Класс построения датасета для модели
class SkipGramDataset(Dataset):
    def __init__(self, pairs, device: torch.device = 'cpu'):
        self.pairs = pairs
        self.device = device
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.LongTensor([center]).to(self.device), torch.LongTensor([context]).to(self.device)

# Класс для ранней остановки при переобучении модели
class EarlyStopping:
    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        if mode not in {'min', 'max'}:
            raise ValueError('Параметр mode может принимать только значения max и min.\n')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('Параметр threshold_mode может принимать только значения rel и abs.\n')
        if not isinstance(patience, int):
            raise ValueError('Параметр patience должен быть целым числом.\n')
        if not isinstance(threshold, float):
            raise ValueError('Параметр threshold должен быть меньше 1,0.\n')
        
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None
        
    def __call__(self, tracked_parameter):
        current = float(tracked_parameter)
        if self.best is None:
            self.best = current
            return False
        
        if self.changed_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1
            
        if self.count >= self.patience:
            self.count = 0
            return True
        return False
    
    def changed_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold
        
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        
        else: # self.mode == 'max' and self.threshold_mode == 'abs'
            return current > best + self.threshold    

# Функция формирования информации о модели
def get_model_info_skipgram_ns(model: SkipGramNS, save_dir_paths: list[str], checkpoint_file_path: str, 
                               len_dataset_train: int, len_dataset_val: int, len_dataset_test: int,
                               device: torch.device, n_samples: int, window_size: int, batch_size: int, epochs: int,
                               graph: bool, continue_flag: bool = False) -> dict:
    for path in save_dir_paths:
        if not os.path.isdir(path):
            cf.print_critical_error(f'Папка {path} не найден.', end='\n\n', prefix='\n')
            exit(1)
            
    model_info = {
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'paths': {
            'checkpoints': save_dir_paths[0],
            'graphs': save_dir_paths[1],
            'checkpoint_file': checkpoint_file_path,
        },
        'model_parameters': {
            name: tuple(param.shape)
            for name, param in model.named_parameters() if param.requires_grad
        },
        'vocab_size': model.vocab_size,
        'embedding': {
            'size': model.embedding_dim,
            'matrix_shape': tuple(model.in_embeddings.weight.shape)
        },
        'training': {
            'device': str(device),
            'format': ('Обучение произведено за один раз' if not continue_flag else 'Ранее обучение было прервано'),
            'graph': graph,
            'samples': n_samples,
            'window_size': window_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'noise_dist': (
                'Равномерное распределение вероятностей выбора негативных примеров'
                if model.noise_dist is None else
                'Кастомное распределение вероятностей выбора негативных примеров'
            ),
            'len_dataset_train': len_dataset_train,
            'len_dataset_val': len_dataset_val
        },
        'testing': {
            'len_dataset_test': len_dataset_test
            },
        'note': 'Модель Word2Vector архитектуры SkipGram с функцией активации Negative Sampling и функцией потерь Binary Cross Entropy'
    }
    
    return model_info

# Функция обучения модели
def train_skipgram_ns(save_dir_paths: list[str], word2idx: dict[str, int], idx2word: list[str], model: SkipGramNS, 
                   dataset_train: SkipGramDataset, dataset_val: SkipGramDataset, dataset_test: SkipGramDataset, device: torch.device = 'cpu') -> None:
    for path in save_dir_paths:
        if not os.path.isdir(path):
            cf.print_critical_error(f'Папка {path} не найдена.', end='\n\n', prefix='\n')
            exit(1)
    
    # Функция потерь
    criterion = NegativeSamplingLoss()
    # Оптимизатор обучения (Adam)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    # Загрузчик данных
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size_w2v_sg_ns, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size_w2v_sg_ns, shuffle=True)
    # Шедулер для изменения скорости обучения
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                              mode = 'min',
                                                              factor = 0.1,
                                                              patience = 4,
                                                              threshold=1e-4,
                                                              threshold_mode='rel')
    # Рання остановка
    earlystopping = EarlyStopping(mode='min', patience=8, threshold=1e-4, threshold_mode='rel')
    
    # Списки
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lr_list = []
    # Переменные
    threshold = 1e-3 # 0.001% - улучшение функции потерь на валидации
    best_loss = None
    
    print_loss_acc_info(config.n_samples_w2v_sg_ns)
    
    cf.print_warn("Запущено обучение модели! Пожалуйста, подождите...", end='\n\n')

    # Обучение
    for epoch in range(config.epochs_w2v_sg_ns):
        # Тренировка модели
        model.train()
        running_train_loss = [] # Вычисление ошибки 
        correct_predictions = 0
        train_loop = tqdm(dataloader_train, leave=False)
        for center_batch, context_batch in train_loop:
            # Данные
            center_batch = center_batch.squeeze().to(device)   # (batch_size,)
            context_batch = context_batch.squeeze().to(device) # (batch_size,)

            # Прямой проход
            input_vectors = model.forward_input(center_batch)                                                # (batch_size, embedding_dim)
            output_vectors = model.forward_output(context_batch)                                             # (batch_size, embedding_dim)
            noise_vectors = model.forward_noise(center_batch.size(0), config.n_samples_w2v_sg_ns).to(device) # (batch_size, n_samples, embedding_dim)

            # Расчет ошибки
            loss = criterion(input_vectors, output_vectors, noise_vectors, device)
            
            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            # Шаг оптимизации
            optimizer.step()

            # Ошибка
            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            # Вычисление метрики
            for i in range(input_vectors.size(0)):                                 # Это даст реальный текущий размер батча
                input_vector = input_vectors[i].unsqueeze(0).to(device)            # (1, embedding_dim)
                output_vector = output_vectors[i].unsqueeze(0).to(device)          # (1, embedding_dim)
                
                # Для контекстных слов
                logit = torch.matmul(input_vector, output_vector.T)                # (1,1)
                prob = torch.sigmoid(logit)
                if prob.item() > 0.5:
                    correct_predictions += 1
                
                # Для негативных примеров
                for j in range(noise_vectors.size(1)):                              # Это итератор по n_samples
                    noise_vector = noise_vectors[i, j].unsqueeze(0).to(device)      # (batch_size, n_samples, embedding_dim) -> (embedding_dim,).unsqueeze(0) -> (1, embedding_dim)
                    noise_logit = torch.matmul(input_vector, noise_vector.T)        # (1,1)
                    noise_prob = torch.sigmoid(noise_logit)
                    if noise_prob.item() < 0.5:
                        correct_predictions += 1    
            
            train_loop.set_description(f"Epoch [{epoch+1}/{config.epochs_w2v_sg_ns}]: train_loss={mean_train_loss:.6f}")
        
        # Расчет значения метрики
        running_train_acc = correct_predictions / (len(dataset_train) * (config.n_samples_w2v_sg_ns + 1))
        
        # Сохранение значения функции потерь и метрики
        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)
        
        # Проверка модели (валидация)
        model.eval()
        with torch.no_grad():
            running_val_loss = [] # Вычисление ошибки 
            correct_predictions = 0
            for center_batch, context_batch in dataloader_val:
                # Данные
                center_batch = center_batch.squeeze().to(device)   # (batch_size,)
                context_batch = context_batch.squeeze().to(device) # (batch_size,)

                # Прямой проход
                input_vectors = model.forward_input(center_batch)                                                # (batch_size, embedding_dim)
                output_vectors = model.forward_output(context_batch)                                             # (batch_size, embedding_dim)
                noise_vectors = model.forward_noise(center_batch.size(0), config.n_samples_w2v_sg_ns).to(device) # (batch_size, n_samples, embedding_dim)

                # Расчет ошибки
                loss = criterion(input_vectors, output_vectors, noise_vectors, device)
                
                # Ошибка
                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss)/len(running_val_loss)
                
                # Вычисление метрики
                for i in range(input_vectors.size(0)):                                 # Это даст реальный текущий размер батча
                    input_vector = input_vectors[i].unsqueeze(0).to(device)            # (1, embedding_dim)
                    output_vector = output_vectors[i].unsqueeze(0).to(device)          # (1, embedding_dim)
                
                    # Для контекстных слов
                    logit = torch.matmul(input_vector, output_vector.T)                # (1,1)
                    prob = torch.sigmoid(logit)
                    if prob.item() > 0.5:
                        correct_predictions += 1
                
                    # Для негативных примеров
                    for j in range(noise_vectors.size(1)):                              # Это итератор по n_samples
                        noise_vector = noise_vectors[i, j].unsqueeze(0).to(device)      # (batch_size, n_samples, embedding_dim) -> (embedding_dim,).unsqueeze(0) -> (1, embedding_dim)
                        noise_logit = torch.matmul(input_vector, noise_vector.T)        # (1,1)
                        noise_prob = torch.sigmoid(noise_logit)
                        if noise_prob.item() < 0.5:
                            correct_predictions += 1    
            
            # Расчет значения метрики
            running_val_acc = correct_predictions / (len(dataset_val) * (config.n_samples_w2v_sg_ns + 1))
            
            # Сохранение значения функции потерь и метрики
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)
        
            lr_scheduler.step(mean_val_loss)
            lr = lr_scheduler.get_last_lr()[0]
            lr_list.append(lr)

            cf.print_info(f"Epoch [{epoch+1}/{config.epochs_w2v_sg_ns}]: train_loss={mean_train_loss:.6f}, train_acc={running_train_acc:.6f}, val_loss={mean_val_loss:.6f}, val_acc={running_val_acc:.6f}, lr={lr:.6f}")
        
            # Сохранение модели после значимого улучшения в процессе обучения (хотя бы на 1%)
            if best_loss is None:
                best_loss = mean_val_loss
            if mean_val_loss < best_loss - best_loss * threshold:
                best_loss = mean_val_loss

                # Текущая эпоха
                save_epoch = epoch
                # Путь сохранения модели
                save_path = os.path.join(save_dir_paths[0], f'W2V_SkipGramNS_model_checkpoint_epoch_{epoch+1}.pt')
                save_path_dataset_train = os.path.join(save_dir_paths[0], config.dataset_train_file_name_w2v_sg_ns)
                save_path_dataset_val = os.path.join(save_dir_paths[0], config.dataset_val_file_name_w2v_sg_ns) 
                save_path_dataset_test = os.path.join(save_dir_paths[0], config.dataset_test_file_name_w2v_sg_ns)
                save_path_word2idx = os.path.join(save_dir_paths[0], config.word2idx_file_name_w2v_sg_ns)
                save_path_idx2word = os.path.join(save_dir_paths[0], config.idx2word_file_name_w2v_sg_ns)
                # Состояние модели
                checkpoint = {
                'info': get_model_info_skipgram_ns(model, save_dir_paths, save_path, 
                                                   len(dataset_train), len(dataset_val), len(dataset_test),
                                                   device=device, n_samples=config.n_samples_w2v_sg_ns, 
                                                   window_size=config.window_size_w2v_sg_ns, batch_size=config.batch_size_w2v_sg_ns, 
                                                   epochs=config.epochs_w2v_sg_ns, graph=config.graph_w2v_sg_ns, continue_flag=False),
                'save_dir_paths': save_dir_paths, 
                'model_params': {
                    'vocab_size': model.vocab_size,
                    'embedding_dim': model.embedding_dim,
                    'noise_dist': model.noise_dist,
                    'n_samples': config.n_samples_w2v_sg_ns,
                    },
                'vocab': {
                    'word2idx_path': save_path_word2idx,
                    'idx2word_path': save_path_idx2word,
                    },
                'dataloader': {
                    'train_dataset_path': save_path_dataset_train,
                    'val_dataset_path': save_path_dataset_val,
                    'test_dataset_path': save_path_dataset_test,
                    'batch_size': config.batch_size_w2v_sg_ns,
                    'shuffle': True,
                    },
                'graph': config.graph_w2v_sg_ns,  
                'earlystopping': {
                     'mode': earlystopping.mode,
                     'patience': earlystopping.patience,
                     'threshold': earlystopping.threshold,
                     'threshold_mode': earlystopping.threshold_mode
                    },
                'state_model': model.state_dict(),
                'state_optimizer': optimizer.state_dict(),
                'state_lr_sheduler': lr_scheduler.state_dict(),
                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss,
                    'threshold': threshold,
                    },
                'metric': {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    },
                'lr': lr_list,
                'epoch': {
                    'EPOCHS': config.epochs_w2v_sg_ns,
                    'save_epoch': save_epoch
                    }
                }
                # Сохранение модели
                torch.save(checkpoint, save_path)
                cf.print_warn(f'На эпохе - {epoch+1}: сохранена модель со значением функции потерь на валидации -', end=' ', prefix='\n'); cf.print_key_info(f'{mean_val_loss:.6f}')
                cf.print_warn('Путь:', end=' '); cf.print_key_info(f'{save_path}')
                # Сохранение данных
                if not os.path.isfile(save_path_dataset_train): 
                    cf.save_to_pickle(dataset_train, save_path_dataset_train)
                    cf.print_info('Тренировочный датасет сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_dataset_train}')
                if not os.path.isfile(save_path_dataset_val): 
                    cf.save_to_pickle(dataset_val, save_path_dataset_val)
                    cf.print_info('Валидационный датасет сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_dataset_val}')
                if not os.path.isfile(save_path_dataset_test): 
                    cf.save_to_pickle(dataset_test, save_path_dataset_test)
                    cf.print_info('Тестирующий датасет сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_dataset_test}')
                if not os.path.isfile(save_path_word2idx): 
                    cf.save_to_json(word2idx, save_path_word2idx)
                    cf.print_info('Словарь word2idx сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_word2idx}')
                if not os.path.isfile(save_path_idx2word): 
                    cf.save_to_json(idx2word, save_path_idx2word)
                    cf.print_info('Словарь idx2word сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_idx2word}')
                print()

            try:
                if earlystopping(mean_val_loss):
                    cf.print_warn(f'Обучение модели остановлено на {epoch+1} эпохе (переобучение)!', end='\n\n')
                    break
            except Exception as e:
                cf.print_critical_error('Невозможно проверить переобучение.', prefix='\n')
                cf.print_critical_error(f'Причина: {e}', end='\n\n')
                exit(1)
        
    # Отображение графиков
    if config.graph_w2v_sg_ns:
        save_dir = save_dir_paths[1]
        # График скорости обучения
        plt.clf()
        plt.title('Learning rate')
        plt.plot(lr_list)
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_learning_rate.png'))
    
        # График значения функции потерь
        plt.clf()
        plt.title('Meaning of the loss function')
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(['loss_train', 'loss_val'])
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_loss_function.png'))
    
        # График значения метрики
        plt.clf()
        plt.title('Meaning of the accuracy')
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.legend(['acc_train', 'acc_val'])
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_accuracy.png'))
        cf.print_info('Графики для анализа скорости обучения, функции потерь и метрик сохранены в папку.')
        cf.print_info('Папка:', end=' '); cf.print_key_info(f'{save_dir}', end='\n\n')
    cf.print_success('Обучение завершено!')

# Функция продолжения обучения модели
def continue_train_skipgram_ns(checkpoint: dict, device: torch.device) -> None:
    # Создание модели
    new_model = SkipGramNS(checkpoint['model_params']['vocab_size'], checkpoint['model_params']['embedding_dim'], 
                       device=device, embedding_weights=None, noise_dist=checkpoint['model_params']['noise_dist']).to(device)
    
    # Функция потерь
    new_criterion = NegativeSamplingLoss()
    # Оптимизатор обучения (Adam)
    new_optimizer = torch.optim.SparseAdam(new_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    # Шедулер для изменения скорости обучения
    new_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer)

    # Загрузка из чекпоинта
    new_model.load_state_dict(checkpoint['state_model'])
    new_optimizer.load_state_dict(checkpoint['state_optimizer'])
    new_lr_scheduler.load_state_dict(checkpoint['state_lr_sheduler'])
    
    # Загрузка из файлов
    train_dataset = cf.load_from_pickle(checkpoint['dataloader']['train_dataset_path'])
    val_dataset = cf.load_from_pickle(checkpoint['dataloader']['val_dataset_path'])
    test_dataset = cf.load_from_pickle(checkpoint['dataloader']['test_dataset_path'])

    # Загрузка из чекпоинта
    dataloader_train = DataLoader(train_dataset, batch_size=checkpoint['dataloader']['batch_size'], 
                                  shuffle=checkpoint['dataloader']['shuffle'])
    dataloader_val = DataLoader(val_dataset, batch_size=checkpoint['dataloader']['batch_size'], 
                                shuffle=checkpoint['dataloader']['shuffle'])
    
    graph = checkpoint['graph']
    earlystopping = EarlyStopping(mode=checkpoint['earlystopping']['mode'], patience=checkpoint['earlystopping']['patience'],
                                  threshold=checkpoint['earlystopping']['threshold'], threshold_mode=checkpoint['earlystopping']['threshold_mode'])
    
    EPOCHS = checkpoint['epoch']['EPOCHS']
    save_epoch = checkpoint['epoch']['save_epoch']

    train_loss = checkpoint['loss']['train_loss']
    val_loss = checkpoint['loss']['val_loss']
    train_acc = checkpoint['metric']['train_acc']
    val_acc = checkpoint['metric']['val_acc']
    lr_list = checkpoint['lr']
    
    threshold = checkpoint['loss']['threshold']
    best_loss = checkpoint['loss']['best_loss']
    
    save_dir_paths = checkpoint['save_dir_paths']
    
    print_loss_acc_info(checkpoint['model_params']['n_samples'])
    
    cf.print_warn('Запущено продолжение обучения модели! Пожалуйста, подождите...', end='\n\n')

    for epoch in range(save_epoch + 1, EPOCHS):
        # Тренировка модели
        new_model.train()
        running_train_loss = [] # Вычисление ошибки 
        correct_predictions = 0
        train_loop = tqdm(dataloader_train, leave=False)
        for center_batch, context_batch in train_loop:
            # Данные
            center_batch = center_batch.squeeze().to(device)   # (batch_size,)
            context_batch = context_batch.squeeze().to(device) # (batch_size,)

            # Прямой проход
            input_vectors = new_model.forward_input(center_batch)                                                             # (batch_size, embedding_dim)
            output_vectors = new_model.forward_output(context_batch)                                                          # (batch_size, embedding_dim)
            noise_vectors = new_model.forward_noise(center_batch.size(0), checkpoint['model_params']['n_samples']).to(device) # (batch_size, n_samples, embedding_dim)

            # Расчет ошибки
            loss = new_criterion(input_vectors, output_vectors, noise_vectors, device)
            
            # Обратный проход
            new_optimizer.zero_grad()
            loss.backward()
            # Шаг оптимизации
            new_optimizer.step()

            # Ошибка
            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            # Вычисление метрики
            for i in range(input_vectors.size(0)):                                 # Это даст реальный текущий размер батча
                input_vector = input_vectors[i].unsqueeze(0).to(device)            # (1, embedding_dim)
                output_vector = output_vectors[i].unsqueeze(0).to(device)          # (1, embedding_dim)
                
                # Для контекстных слов
                logit = torch.matmul(input_vector, output_vector.T)                # (1,1)
                prob = torch.sigmoid(logit)
                if prob.item() > 0.5:
                    correct_predictions += 1
                
                # Для негативных примеров
                for j in range(noise_vectors.size(1)):                              # Это итератор по n_samples
                    noise_vector = noise_vectors[i, j].unsqueeze(0).to(device)      # (batch_size, n_samples, embedding_dim) -> (embedding_dim,).unsqueeze(0) -> (1, embedding_dim)
                    noise_logit = torch.matmul(input_vector, noise_vector.T)        # (1,1)
                    noise_prob = torch.sigmoid(noise_logit)
                    if noise_prob.item() < 0.5:
                        correct_predictions += 1    
            
            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]: train_loss={mean_train_loss:.6f}")
        
        # Расчет значения метрики
        running_train_acc = correct_predictions / (len(train_dataset) * (checkpoint['model_params']['n_samples'] + 1))
        
        # Сохранение значения функции потерь и метрики
        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)
        
        # Проверка модели (валидация)
        new_model.eval()
        with torch.no_grad():
            running_val_loss = [] # Вычисление ошибки 
            correct_predictions = 0
            for center_batch, context_batch in dataloader_val:
                # Данные
                center_batch = center_batch.squeeze().to(device)   # (batch_size,)
                context_batch = context_batch.squeeze().to(device) # (batch_size,)

                # Прямой проход
                input_vectors = new_model.forward_input(center_batch)                                                             # (batch_size, embedding_dim)
                output_vectors = new_model.forward_output(context_batch)                                                          # (batch_size, embedding_dim)
                noise_vectors = new_model.forward_noise(center_batch.size(0), checkpoint['model_params']['n_samples']).to(device) # (batch_size, n_samples, embedding_dim)

                # Расчет ошибки
                loss = new_criterion(input_vectors, output_vectors, noise_vectors, device)
                
                # Ошибка
                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss)/len(running_val_loss)
                
                # Вычисление метрики
                for i in range(input_vectors.size(0)):                                 # Это даст реальный текущий размер батча
                    input_vector = input_vectors[i].unsqueeze(0).to(device)            # (1, embedding_dim)
                    output_vector = output_vectors[i].unsqueeze(0).to(device)          # (1, embedding_dim)
                
                    # Для контекстных слов
                    logit = torch.matmul(input_vector, output_vector.T)                # (1,1)
                    prob = torch.sigmoid(logit)
                    if prob.item() > 0.5:
                        correct_predictions += 1
                
                    # Для негативных примеров
                    for j in range(noise_vectors.size(1)):                              # Это итератор по n_samples
                        noise_vector = noise_vectors[i, j].unsqueeze(0).to(device)      # (batch_size, n_samples, embedding_dim) -> (embedding_dim,).unsqueeze(0) -> (1, embedding_dim)
                        noise_logit = torch.matmul(input_vector, noise_vector.T)        # (1,1)
                        noise_prob = torch.sigmoid(noise_logit)
                        if noise_prob.item() < 0.5:
                            correct_predictions += 1    
            
            # Расчет значения метрики
            running_val_acc = correct_predictions / (len(val_dataset) * (checkpoint['model_params']['n_samples'] + 1))
            
            # Сохранение значения функции потерь и метрики
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)
        
            new_lr_scheduler.step(mean_val_loss)
            lr = new_lr_scheduler.get_last_lr()[0]
            lr_list.append(lr)

            cf.print_info(f"Epoch [{epoch+1}/{EPOCHS}]: train_loss={mean_train_loss:.6f}, train_acc={running_train_acc:.6f}, val_loss={mean_val_loss:.6f}, val_acc={running_val_acc:.6f}, lr={lr:.6f}")
        
            # Сохранение модели после значимого улучшения в процессе обучения (хотя бы на 1%)
            if best_loss is None:
                best_loss = mean_val_loss
            if mean_val_loss < best_loss - best_loss * threshold:
                best_loss = mean_val_loss

                # Текущая эпоха
                save_epoch = epoch
                # Путь сохранения модели
                save_path = os.path.join(save_dir_paths[0], f'W2V_SkipGramNS_model_checkpoint_epoch_{epoch+1}.pt')
                # Состояние модели
                new_checkpoint = {
                'info': get_model_info_skipgram_ns(new_model, save_dir_paths, save_path, 
                                                   len(train_dataset), len(val_dataset), len(test_dataset),
                                                   device=device, n_samples=checkpoint['info']['training']['samples'],
                                                   window_size=checkpoint['info']['training']['window_size'], batch_size=checkpoint['info']['training']['batch_size'],
                                                   epochs=checkpoint['info']['training']['epochs'], graph=checkpoint['info']['training']['graph'], continue_flag=True),
                'save_dir_paths': save_dir_paths, 
                'model_params': {
                    'vocab_size': new_model.vocab_size,
                    'embedding_dim': new_model.embedding_dim,
                    'noise_dist': new_model.noise_dist,
                    'n_samples': checkpoint['model_params']['n_samples'],
                    },
                'vocab': {
                    'word2idx_path': checkpoint['vocab']['word2idx_path'],
                    'idx2word_path': checkpoint['vocab']['idx2word_path'],
                    },
                'dataloader': {
                    'train_dataset_path': checkpoint['dataloader']['train_dataset_path'],
                    'val_dataset_path': checkpoint['dataloader']['val_dataset_path'],
                    'test_dataset_path': checkpoint['dataloader']['test_dataset_path'],
                    'batch_size': checkpoint['dataloader']['batch_size'],
                    'shuffle': True,
                    },
                'graph': graph,  
                'earlystopping': {
                     'mode': earlystopping.mode,
                     'patience': earlystopping.patience,
                     'threshold': earlystopping.threshold,
                     'threshold_mode': earlystopping.threshold_mode
                    },
                'state_model': new_model.state_dict(),
                'state_optimizer': new_optimizer.state_dict(),
                'state_lr_sheduler': new_lr_scheduler.state_dict(),
                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss,
                    'threshold': threshold,
                    },
                'metric': {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    },
                'lr': lr_list,
                'epoch': {
                    'EPOCHS': EPOCHS,
                    'save_epoch': save_epoch
                    }
                }
                # Сохранение модели
                torch.save(new_checkpoint, save_path)
                cf.print_warn(f'На эпохе - {epoch+1}: сохранена модель со значением функции потерь на валидации -', end=' ', prefix='\n'); cf.print_key_info(f'{mean_val_loss:.6f}')
                cf.print_warn('Путь:', end=' '); cf.print_key_info(f'{save_path}', end='\n\n')
            try:
                if earlystopping(mean_val_loss):
                    cf.print_warn(f'Обучение модели остановлено на {epoch+1} эпохе (переобучение)!', end='\n\n')
                    break
            except Exception as e:
                cf.print_critical_error('Невозможно проверить переобучение.', prefix='\n')
                cf.print_critical_error(f'Причина: {e}', end='\n\n')
                exit(1)
        
    # Отображение графиков
    if graph:
        save_dir = save_dir_paths[1]
        # График скорости обучения
        plt.clf()
        plt.title('Learning rate')
        plt.plot(lr_list)
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_learning_rate.png'))
    
        # График значения функции потерь
        plt.clf()
        plt.title('Meaning of the loss function')
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(['loss_train', 'loss_val'])
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_loss_function.png'))
    
        # График значения метрики
        plt.clf()
        plt.title('Meaning of the accuracy')
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.legend(['acc_train', 'acc_val'])
        plt.savefig(os.path.join(save_dir, 'W2V_SkipGramNS_accuracy.png'))
        cf.print_info('Графики для анализа скорости обучения, функции потерь и метрик сохранены в папку.')
        cf.print_info('Папка:', end=' '); cf.print_key_info(f'{save_dir}', end='\n\n')
    cf.print_success('Продолженное обучение модели завершено!')
    
# Функция тестирования модели
def test_skipgram_ns(model: SkipGramNS, dataset_test: SkipGramDataset, n_samples: int, device: torch.device = 'cpu') -> None:
    # Функция потерь
    criterion = NegativeSamplingLoss()
    # Загрузчик данных
    dataloader_test = DataLoader(dataset_test, batch_size=config.test_batch_size_w2v_sg_ns, shuffle=True)
    
    print_loss_acc_info(n_samples)

    cf.print_warn('Запущено тестирование модели! Пожалуйста, подождите...', end='\n\n')

    # Проверка модели (валидация)
    model.eval()
    with torch.no_grad():
        running_test_loss = [] # Вычисление ошибки 
        correct_predictions = 0
        test_loop = tqdm(dataloader_test, leave=False)
        for center_batch, context_batch in test_loop:
            # Данные
            center_batch = center_batch.squeeze().to(device)   # (batch_size,)
            context_batch = context_batch.squeeze().to(device) # (batch_size,)

            # Прямой проход
            input_vectors = model.forward_input(center_batch)                                                # (batch_size, embedding_dim)
            output_vectors = model.forward_output(context_batch)                                             # (batch_size, embedding_dim)
            noise_vectors = model.forward_noise(center_batch.size(0), n_samples=n_samples).to(device)        # (batch_size, n_samples, embedding_dim)

            # Расчет ошибки
            loss = criterion(input_vectors, output_vectors, noise_vectors, device)
                
            # Ошибка
            running_test_loss.append(loss.item())
            mean_test_loss = sum(running_test_loss)/len(running_test_loss)
                
            # Вычисление метрики
            for i in range(input_vectors.size(0)):                                 # Это даст реальный текущий размер батча
                input_vector = input_vectors[i].unsqueeze(0).to(device)            # (1, embedding_dim)
                output_vector = output_vectors[i].unsqueeze(0).to(device)          # (1, embedding_dim)
                
                # Для контекстных слов
                logit = torch.matmul(input_vector, output_vector.T)                # (1,1)
                prob = torch.sigmoid(logit)
                if prob.item() > 0.5:
                    correct_predictions += 1
                
                # Для негативных примеров
                for j in range(noise_vectors.size(1)):                              # Это итератор по n_samples
                    noise_vector = noise_vectors[i, j].unsqueeze(0).to(device)      # (batch_size, n_samples, embedding_dim) -> (embedding_dim,).unsqueeze(0) -> (1, embedding_dim)
                    noise_logit = torch.matmul(input_vector, noise_vector.T)        # (1,1)
                    noise_prob = torch.sigmoid(noise_logit)
                    if noise_prob.item() < 0.5:
                        correct_predictions += 1   
                        
            test_loop.set_description(f"test_loss={mean_test_loss:.6f}")
            
        # Расчет значения метрики
        running_test_acc = correct_predictions / (len(dataset_test) * (n_samples + 1))
            
    cf.print_info("Результаты на тестовой выборке:")
    cf.print_info(f"Значение функции потерь: ", end=""); cf.print_key_info(f"{mean_test_loss:.6f}", end=", "); print("значение метрики: ", end=""); cf.print_key_info(f"{running_test_acc:.6f}", end='\n\n')

    cf.print_success('Тестирование завершено!')