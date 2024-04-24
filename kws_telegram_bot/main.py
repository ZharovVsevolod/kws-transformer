# -*- coding: utf-8 -*-
from aiogram import Bot, Dispatcher, executor, types
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import uuid
import os


API_KEY = "..." # your telefram bot api_key


bot = Bot(API_KEY)
dp = Dispatcher(bot)

main_commands = ["Быстрее", "Медленнее", "Налево", "Направо", "Вперед", "Назад", "Бегом", "Стоп", "Привет", "Ура"]
noise_commands = ["Клюшка", "Голос", "Макароны", "Лампочка", "Закат", "Рука", "Величие", "Носок", "Бред", "Фонарь"]


# # Авторизация
# gauth = GoogleAuth()
# if gauth.credentials is None:
#     # Аутентификация через credentials.json
#     gauth.LocalWebserverAuth()
# elif gauth.access_token_expired:
#     # Обновление токена, если он устарел
#     gauth.Refresh()
# else:
#     # Авторизация с сохранением токена
#     gauth.Authorize()
# # Создание объекта GoogleDrive
# drive = GoogleDrive(gauth)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    desc = 'Привет!\nЯ бот, который нужен для сбора данных для проекта по распознованию слов.\n\n\
Запись должна проходить в тихом месте без посторонних шумов. Микрофон следует держать на расстоянии не менее 20 см от рта\n\n\
Сначала появится снизу клавиатура для выбора слова. Выберите слово, которое будете говорить. Затем скажите его и отправьте голосовое сообщение (1-3 секунды).\n\n\
Приветствуется - разные интонации сказанного и выбор слова больше одного раза.'
    
    kb_main = [
        [
             types.KeyboardButton('Шум (список иных слов)'),
         ],
        [
            types.KeyboardButton('Быстрее'),
            types.KeyboardButton('Медленнее'),
         ],
        [
            types.KeyboardButton('Налево'),
            types.KeyboardButton('Направо')
        ],
        [
            types.KeyboardButton('Вперед'),
            types.KeyboardButton('Назад')
        ],
        [
            types.KeyboardButton('Бегом'),
            types.KeyboardButton('Стоп')
        ],
        [
            types.KeyboardButton('Привет'),
            types.KeyboardButton('Ура')
        ],
    ]

    markup = types.ReplyKeyboardMarkup(keyboard=kb_main, one_time_keyboard=True, resize_keyboard=True, input_field_placeholder="Выберите слово для голосового сообщения")
    await message.reply(desc, reply_markup=markup)


@dp.message_handler(lambda message: message.text == "Шум (список иных слов)")
async def without_puree(message: types.Message):
    desc="Приведен список слов для инициализации шума в клавиатуре, которые позволят обучить лучше модель для распознования слов"
    
    kb_noise = [
    [
        types.KeyboardButton('Вернуться к основным словам'),
    ],
    [
        types.KeyboardButton('Клюшка'),
        types.KeyboardButton('Голос'),
    ],
    [
        types.KeyboardButton('Макароны'),
        types.KeyboardButton('Лампочка')
    ],
    [
        types.KeyboardButton('Закат'),
        types.KeyboardButton('Рука')
    ],
    [
        types.KeyboardButton('Величие'),
        types.KeyboardButton('Носок')
    ],
    [
        types.KeyboardButton('Бред'),
        types.KeyboardButton('Фонарь'),
    ],
    ]
    
    markup = types.ReplyKeyboardMarkup(keyboard=kb_noise, one_time_keyboard=True, resize_keyboard=True, input_field_placeholder="Выберите слово для голосового сообщения")
    await message.reply(desc, reply_markup=markup)


@dp.message_handler(lambda message: message.text == "Вернуться к основным словам")
async def without_puree(message: types.Message):
    await start(message)


@dp.message_handler(lambda message: message.text in main_commands or message.text in noise_commands)
async def without_puree(message: types.Message):
    await message.reply(f"Теперь выберите слово выше, нажав ответить/reply и скажите в голосовом сообщении: {message.text}")


@dp.message_handler(content_types=['voice'])
async def voice_message(message: types.Message):
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    
    temp_message = message.reply_to_message
    print(temp_message)
    
    if temp_message.text in main_commands or temp_message.text in noise_commands:
        # for local
        name = temp_message.text
        if temp_message.text in noise_commands:
             dir = "./data/noise/" + name
        else:
            dir = "./data/" + name
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        
        # # Название основной папки
        # main_folder_name = "data"
        # # Проверка существования основной папки
        # main_folder_list = drive.ListFile({'q': "title='" + main_folder_name + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
        # if main_folder_list:
        #     main_folder = main_folder_list[0]
        # else:
        #     # Создание основной папки на Google Drive
        #     main_folder = drive.CreateFile({'title': main_folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
        #     main_folder.Upload()
        # # Получение ID основной папки
        # main_folder_id = main_folder['id']
        
        
        # sub_folder_name = ""
        # if temp_message.text in noise_commands:
        #     sub_folder_name = "noise"
        #     # Проверка существования вложенной папки в основной папке
        #     sub_folder_list = drive.ListFile({'q': "title='" + sub_folder_name + "' and '" + main_folder_id + "' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
        #     if sub_folder_list:
        #         sub_folder = sub_folder_list[0]
        #     else:
        #         # Создание вложенной папки в основной папке на Google Drive
        #         sub_folder = drive.CreateFile({'title': sub_folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': main_folder_id}]})
        #         sub_folder.Upload()
        #     main_folder_id = sub_folder['id']
        
        # # Название вложенной папки
        # sub_folder_name = f"{temp_message.text}"

        # # Проверка существования вложенной папки в основной папке
        # sub_folder_list = drive.ListFile({'q': "title='" + sub_folder_name + "' and '" + main_folder_id + "' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
        # if sub_folder_list:
        #     sub_folder = sub_folder_list[0]
        # else:
        #     # Создание вложенной папки в основной папке на Google Drive
        #     sub_folder = drive.CreateFile({'title': sub_folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': main_folder_id}]})
        #     sub_folder.Upload()

        # # Получение ID вложенной папки
        # sub_folder_id = sub_folder['id']
        
        
        # file_name = f"{str(uuid.uuid4())[0:17]}.wav"
        # file = drive.CreateFile({'title': file_name, 'parents': [{'id': sub_folder_id}]})
        # file.Upload()
        
        
        # for local:
        await bot.download_file(file_path, f"{dir}/{str(uuid.uuid4())[0:17]}.wav")
        await bot.send_message(message.chat.id, 'Сообщение получено и сохранено успешно.')
        await bot.send_message(message.chat.id, 'Отлично!\nТеперь повторите либо с этим же словом, либо с другим (клавиатура может быть скрыта).')
    else:
        await bot.send_message(message.chat.id, "Значение некорректное, введите еще раз /start")


@dp.callback_query_handler()
async def callback(call):
    print(call.data)
    await call.message.answer(call.data)


if __name__ == "__main__":
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        print(e)