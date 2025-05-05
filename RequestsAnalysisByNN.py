import core_functions as cf
import time, os

def clear_console():
    # Для Windows
    if os.name == 'nt':
        os.system('cls')
    # Для Mac и Linux
    else:
        os.system('clear')

# Запуск программы
if __name__ == "__main__":
    """ Вариант запуска"""
    while True:
        time.sleep(1)
        clear_console() # Очистка консоли
        cf.print_welcome()
        mode = input("Введите режим (1 - Разработчик, 2 - Пользователь, 0 - Выход): ").strip()
        # Выбор
        if mode == '1':
            cf.print_success("Вы выбрали режим 'Разработчик': обучение, тестирование, сохранение нейросетей.")
            time.sleep(1)
            cf.print_main_line()
            cf.dev_running_program()
        elif mode == '2':
            cf.print_success("Вы выбрали режим 'Пользователь': обращение -> кластер.")
            time.sleep(1)
            cf.print_main_line()
            cf.user_running_program()
        elif mode == '0':
            cf.print_success("Выход из программы.", end='\n\n')
            exit()
        else:
            cf.print_error("Неверный ввод. Пожалуйста, попробуйте снова.")