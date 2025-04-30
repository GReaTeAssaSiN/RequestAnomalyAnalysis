import numpy as np
import torch, datetime, os
import core_functions as cf
from core_functions import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SOM:
    def __init__(self, x_size: int=1, y_size: int=1, input_dim: int=300, map_flag: bool=False,
                 device: torch.device='cpu', sigma: float=None, learning_rate: float=0.3,
                 num_epochs: int=100) -> None:
        # ШАГ 1. Инициализация сети
        self.x_size = x_size                    # Строки решетки
        self.y_size = y_size                    # Столбцы решетки
        self.input_dim = input_dim              # n-мерное пространство
        self.learning_rate = learning_rate      # Скорость обучения
        self.num_epochs = num_epochs            # Количество эпох
        self.map_flag = map_flag
        self.device = device
        
        self.sigma = sigma if sigma is not None else max(x_size, y_size) / 2                        # Радиус влияния на соседей
        self.weights = torch.rand((x_size, y_size, input_dim), dtype=torch.float32).to(self.device) # Веса нейронов, Tensor(x_size, y_size, input_dim)
        self.weights[0, 0] = torch.zeros(self.input_dim, dtype=torch.float32).to(self.device)

        # Координаты нейронов в решетке; Tensor(x_size, y_size, 2)
        self.neuron_locations = torch.Tensor(
            [[(x,y)] for y in range(y_size) for x in range(x_size)]    
        ).view(x_size, y_size, 2).to(self.device)
    
    def _load_som(self, checkpoint: dict) -> None:
        self.x_size = checkpoint['model_parameters']['x_size']
        self.y_size = checkpoint['model_parameters']['y_size']
        self.input_dim = checkpoint['model_parameters']['input_dim']
        self.learning_rate = checkpoint['model_parameters']['lr']
        self.sigma = checkpoint['model_parameters']['sigma']
        self.weights = checkpoint['model_parameters']['weights'].to(self.device)
        self.neuron_locations = checkpoint['model_parameters']['neuron_locations'].to(self.device)
        self.num_epochs = checkpoint['model_parameters']['num_epochs']
        self.map_flag = checkpoint['model_parameters']['map_flag']

    def _save_som(self, saved_epoch: int, save_dir_paths: list[str], save_path: str, data_path: str, 
                  continue_flag: bool=False) -> None:
        checkpoint = {
            'info': {
                'about': 'Сеть Кохонена для кластеризации обращений потребителей',
                'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'saved_epoch': saved_epoch,
                'device': self.device
                },
            'paths': {
                'checkpoints': save_dir_paths[0],
                'map': save_dir_paths[1],
                'checkpoint_file': save_path,
                'data_path': data_path
                },
            'model_parameters': {
                'x_size': self.x_size,
                'y_size': self.y_size,
                'input_dim': self.input_dim,
                'lr': self.learning_rate,
                'sigma': self.sigma,
                'weights': self.weights,
                'neuron_locations': self.neuron_locations,
                'num_epochs': self.num_epochs,
                'map_flag': self.map_flag
                },
            'training': {
                'format': ('Обучение произведено за один раз' if not continue_flag else 'Ранее обучение было прервано'),
                }
            }
        torch.save(checkpoint, save_path)

    def _find_wta(self, input_vec: torch.Tensor) -> torch.Tensor:
        if torch.count_nonzero(input_vec) == 0:
            # Если вектор полностью нулевой — кластер (0, 0)
            return torch.tensor([0, 0], device=self.device)

        diff = self.weights - input_vec         # broadcasting: input_vec(input_dim) -> input_vec(x_size, y_size, input_dim) повторяется; 
                                                # Tensor(x_size, y_size, input_dim))
        distances = torch.sum(diff ** 2, dim=2) # Евклидово расстояние по input_dim; Tensor(x_size, y_size)
        distances[0, 0] = float('inf')
        
        wta_index = torch.argmin(distances)                                                                # Целое число относительно одномерного вектора
        wta_coords = torch.tensor([wta_index // self.y_size, wta_index % self.y_size], device=self.device) # Преобразование относительно решетки, например,
                                                                                                           # 5 для решетки (3x7) -> (0,5); # Tensor(2,)
        return wta_coords
    
    def _decay_parameter(self, initial_value: float, current_epoch: int, num_epochs: int) -> torch.Tensor:
        initial_value = torch.tensor(initial_value, dtype=torch.float32, device=self.device)
        current_epoch = torch.tensor(current_epoch, dtype=torch.float32, device=self.device)
        num_epochs = torch.tensor(num_epochs, dtype=torch.float32, device=self.device)
        return initial_value * torch.exp(-current_epoch/num_epochs)
    
    def _calculate_influence(self, wta_coords: torch.Tensor, current_sigma: float) -> torch.Tensor:
        d = self.neuron_locations - wta_coords                          # broadcasting; Tensor(x_size, y_size, 2)
        distance_sq = torch.sum(d.float() ** 2, dim=2)                  # Евклидово расстояние; Tensor(x_size, y_size)
        influence = torch.exp(-distance_sq / (2 * current_sigma ** 2))  # Гауссовская функция соседства; Tensor(x_size, y_size)
        return influence

    def train(self, data: np.ndarray, save_dir_paths: list[str], count_save_epoch: int=10) -> None:
        cf.print_warn('Запущено обучение модели! Пожалуйста, подождите...')
        cf.print_info('Количество эпох для сохранения чекпоинта:', end=' '); cf.print_key_info(f'{count_save_epoch}')
        cf.print_warn('Чекпоинты сохраняются в папку:', end=' '); cf.print_key_info(f'{save_dir_paths[0]}', end='\n\n')

        # Сохранение датасета
        save_path_dataset_train = os.path.join(save_dir_paths[0], config.dataset_train_file_name_kohonen)
        if not os.path.isfile(save_path_dataset_train):
            cf.save_to_pickle(data, save_path_dataset_train)
            cf.print_warn('Тренировочный датасет сохранен в файл:', end=' '); cf.print_key_info(f'{save_path_dataset_train}', end='\n\n')

        # Обучение
        epoch_loop = tqdm(range(config.epochs_kohonen), desc='Обучение SOM', unit='epoch')
        for epoch in epoch_loop:
            # Перемешиваем данные
            if isinstance(data, torch.Tensor): 
                data = data.cpu().numpy()
            np.random.shuffle(data)
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            # ШАГ 2. Предъявление сети нового входного примера
            for input_vec in data:
                # ШАГ 3-4. Вычисление расстояния до всех нейронов сети и определение нейрона победителя
                wta_coords = self._find_wta(input_vec)
                
                # Уменьньшение скорости обучения и радиуса соседства
                lr = self._decay_parameter(self.learning_rate, epoch, self.num_epochs)
                sigma = self._decay_parameter(self.sigma, epoch, self.num_epochs)
                # Вычисление влияния соседства на веса нейронов
                influence = self._calculate_influence(wta_coords, sigma)  # shape (x, y)

                # ШАГ 5. Обновление весов нейрона победителя и его соседей
                for x in range(self.x_size):
                    for y in range(self.y_size):
                        if x == 0 and y == 0:
                            continue
                        influence_factor = influence[x, y]
                        self.weights[x, y] += lr * influence_factor * (input_vec - self.weights[x, y])
             
            if (epoch + 1) % count_save_epoch == 0 or (epoch + 1) == self.num_epochs:
                # Сохранение чекпоинта
                save_path = os.path.join(save_dir_paths[0], f'Kohonen_model_checkpoint_epoch_{epoch+1}.pt')
                self._save_som(epoch, save_dir_paths, save_path, save_path_dataset_train, continue_flag=False)

        cf.print_success('Обучение завершено!', prefix='\n')
        
    def continue_train(self, checkpoint: dict, device: torch.device, count_save_epoch: int=10) -> None:
        cf.print_warn('Запущено дообучение модели! Пожалуйста, подождите...')
        cf.print_info('Количество эпох для сохранения чекпоинта:', end=' '); cf.print_key_info(f'{count_save_epoch}')
        cf.print_warn('Чекпоинты сохраняются в папку:', end=' '); cf.print_key_info(f"{checkpoint['paths']['checkpoints']}", end='\n\n')
        
        # Устройство
        self.device = device

        # Загрузка датасета
        data = cf.load_from_pickle(checkpoint['paths']['data_path'])
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
       
        # Загрузка параметров модели
        self._load_som(checkpoint)

        # Добучение
        saved_epoch = checkpoint['info']['saved_epoch']
        EPOCHS = checkpoint['model_parameters']['num_epochs']
        epoch_loop = tqdm(range(saved_epoch + 1, EPOCHS), desc='Обучение SOM', unit='epoch')
        for epoch in epoch_loop:
            # Перемешиваем данные
            if isinstance(data, torch.Tensor): 
                data = data.cpu().numpy()
            np.random.shuffle(data)
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            # ШАГ 2. Предъявление сети нового входного примера
            for input_vec in data:
                # ШАГ 3-4. Вычисление расстояния до всех нейронов сети и определение нейрона победителя
                wta_coords = self._find_wta(input_vec)
                
                # Уменьньшение скорости обучения и радиуса соседства
                lr = self._decay_parameter(self.learning_rate, epoch, self.num_epochs)
                sigma = self._decay_parameter(self.sigma, epoch, self.num_epochs)
                # Вычисление влияния соседства на веса нейронов
                influence = self._calculate_influence(wta_coords, sigma)  # shape (x, y)

                # ШАГ 5. Обновление весов нейрона победителя и его соседей
                for x in range(self.x_size):
                    for y in range(self.y_size):
                        if x == 0 and y == 0:
                            continue
                        influence_factor = influence[x, y]
                        self.weights[x, y] += lr * influence_factor * (input_vec - self.weights[x, y])
             
            if (epoch + 1) % count_save_epoch == 0 or (epoch + 1) == self.num_epochs:
                # Сохранение чекпоинта
                save_path = os.path.join(checkpoint['paths']['checkpoints'], f'Kohonen_model_checkpoint_epoch_{epoch+1}.pt')
                save_dir_paths = [checkpoint['paths']['checkpoints'], checkpoint['paths']['map']]
                self._save_som(epoch, save_dir_paths, save_path, checkpoint['paths']['data_path'], continue_flag=True)

        cf.print_success('Продолженное бучение завершено!', prefix='\n')

    def load_som_by_checkpoint(self, checkpoint: dict, device: torch.device = 'cpu'):
        self.device = device
        self._load_som(checkpoint)
        
    def assign_clusters(self, data: np.ndarray) -> list[tuple[int, int]]:
        self.weights = self.weights.to(self.device)
        if isinstance(data, torch.Tensor):
            data = data.cpu.numpy()
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        cluster_assignments = []
        for input_vec in data:
            wta_coords = self._find_wta(input_vec)
            cluster_assignments.append((int(wta_coords[0].item()), int(wta_coords[1].item())))
        return cluster_assignments
    
    def visualize_clusters(self, save_dir: str, cluster_assignments: list[tuple[int, int]], new_map_flag: bool, title: str='Карта SOM') -> None:
        # Проверка созданной карты
        save_file_path = os.path.join(save_dir, 'Kohonen_map.png')
        if (os.path.isfile(save_file_path)):
            return None
        
        # Визуализация
        if not (self.map_flag or new_map_flag):
            return None

        # Матрица с количеством обращений в каждом кластере
        heatmap = torch.zeros((self.x_size, self.y_size), dtype=torch.int32)
        for x, y in cluster_assignments:
            heatmap[x, y] += 1

        # Построение SOM-карты
        plt.clf()    
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap.cpu().numpy(), annot=True, fmt='d', cmap='YlGnBu')
        plt.title(title)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.tight_layout()
        # Сохранение SOM-карты
        plt.savefig(save_file_path)
        cf.print_info('SOM-карта была сохранена в папку.')
        cf.print_info('Папка:', end=' '); cf.print_key_info(f'{save_dir}')
