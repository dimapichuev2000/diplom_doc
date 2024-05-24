import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from joblib import load
import PyPDF2
import re
import nltk
import string
import shutil
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")


class PDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF File Uploader")
        self.root.geometry("600x400")

        self.file_paths = []  # Переменная для хранения путей к выбранным файлам
        self.text_list = []  # Переменная для хранения text

        # Настройка стиля
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TFrame', background='#f0f0f0')

        self.main_frame = ttk.Frame(root, padding=(20, 20, 20, 20))
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.upload_button = ttk.Button(self.main_frame, text="Upload PDFs", command=self.upload_files)
        self.upload_button.pack(pady=10)
        self.filter_button = ttk.Button(self.button_frame, text="Filter", command=self.filter_files)
        self.filter_button.pack(side=tk.LEFT, padx=5)

        self.text_widget = tk.Text(self.main_frame, height=15)
        self.text_widget.pack(expand=True, fill=tk.BOTH, pady=10)

        self.scrollbar = ttk.Scrollbar(self.text_widget)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.text_widget.yview)

    def upload_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if file_paths:
            self.file_paths = file_paths
            self.text_widget.delete(1.0, tk.END)
            for file_path in self.file_paths:
                file_name = file_path.split('/')[-1]
                self.text_widget.insert(tk.END, f"{file_name}\n")

    def convert_pdf_to_txt(self, pdf_path):
        """
        Конвертирует файл формата PDF в текстовый файл формата TXT
        """
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                text += "\n".join(line.strip() for line in page_text.split("\n") if line.strip())
        return text

    # Функция для чтения текста из файла и классификации его
    def classify_text_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def clean_text(self, text):
        text = re.sub('\n', " ", text)
        text = re.sub(r'[^а-яА-Я]', " ", text)
        text = re.sub(r'\s+', " ", text)
        text = text.lower()
        text = text.split()
        text = [j for j in text if len(text) > 1]
        text = [i for i in text if not i in set(stopwords.words("russian"))]
        text = " ".join(text)
        return text

    def folderDistribution(self, probabilities):
        index = 0
        for probabilitie in probabilities:
            index = index + 1
            for category, prob in enumerate(probabilitie):
                percent = prob * 100
                if percent > 30.0:
                    self.distribute_file(index, category, 'data/')

    def distribute_file(self, index, category, base_path):
        # Словарь с категориями и именами папок
        folders = {
            0: "Политика",
            1: "Спорт",
            2: "Технологии",
            3: "Развлечение",
            4: "Бизнес",
            5: "Юридические цитаты",
            6: "Медицина"
        }

        # Проверяем, что категория валидна
        if category not in folders:
            print(f"Неправильная категория: {category}")
            return
        file_path = self.file_paths[index-1]
        # Имя папки для данной категории
        folder_name = folders[category]

        # Полный путь к папке назначения
        destination_folder = os.path.join(base_path, folder_name)

        # Создаем папку, если она не существует
        os.makedirs(destination_folder, exist_ok=True)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        # Определяем конечный путь к файлу
        destination_path = os.path.join(destination_folder, file_name)
        counter = 1

        # Если файл с таким именем уже существует, добавляем цифру в скобках
        while os.path.exists(destination_path):
            new_file_name = f"{name}({counter}){ext}"
            destination_path = os.path.join(destination_folder, new_file_name)
            counter += 1

        # Перемещаем файл в папку назначения
        shutil.copy(file_path, destination_path)
        print(f"Файл {file_path} перемещен в {destination_path}")

    def filter_files(self):
        self.text_widget.delete(1.0, tk.END)
        self.text_list.clear()
        for file_path in self.file_paths:
            file_name = file_path.split('/')[-1]
            # Предположим, что процентная часть будет заполняться позже
            percentage = "0%"
            self.text_widget.insert(tk.END, f"{file_name}\t\t{percentage}\n")

        for file_path in self.file_paths:
            # Путь к входному файлу
            input_file_path = file_path

            # Получаем расширение файла
            _, file_extension = os.path.splitext(input_file_path)
            text = self.convert_pdf_to_txt(input_file_path)
            # print(text)
            # Проверяем формат файла и вызываем соответствующую функцию конвертации
            if file_extension.lower() == '.pdf':
                self.text_list.append(text)
            else:
                print("Формат файла не поддерживается.")

        model = load('model.pkl')
        vectorizer = load('tfidf_vectorizer.pkl')
        scaler = load('scaler.pkl')

        # Подготовьте новый текст для классификации

        data = {'Text': self.text_list}
        df = pd.DataFrame(data)
        # df = df.drop_duplicates()

        df["cleaned_text"] = df["Text"].apply(self.clean_text)

        df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

        df["text_length"] = df["cleaned_text"].apply(lambda x: len(str(x)))

        df["stopwords_count"] = df["Text"].apply(
            lambda x: len([i for i in x.split() if i in set(stopwords.words("russian"))]))

        df["punct_count"] = df["Text"].apply(lambda x: len([i for i in x if i in string.punctuation]))

        df["caps_count"] = df["Text"].apply(lambda x: len([i for i in str(x) if i.isupper()]))

        df.head()
        count_array = vectorizer.transform(df["cleaned_text"]).toarray()
        df_count_vec = pd.DataFrame(count_array, columns=vectorizer.get_feature_names_out())
        df_count_vec = df_count_vec.reset_index(drop=True)
        df_count_vec.head()
        df1 = df.iloc[:, [2, 3, 4, 5, 6]]
        df1 = df1.reset_index(drop=True)
        df1.head()
        df_nlp = pd.concat([df1, df_count_vec], axis=1)
        df_nlp.head()
        df_scaler = scaler.transform(df_nlp)
        predicted_class = model.predict(df_scaler)
        print(predicted_class)
        probabilities = model.predict_proba(df_scaler)
        self.folderDistribution(probabilities)


def main():
    root = tk.Tk()
    app = PDFApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
