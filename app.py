import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
import joblib
import fitz  # PyMuPDF
import random

# Установка заголовка страницы
st.set_page_config(
    page_title="Классификатор резюме",
    page_icon="📄",
    layout="wide"
)

# Функция для очистки текста
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)  # удаление пунктуации
    text = re.sub(r'\d+', ' ', text)      # удаление цифр
    text = re.sub(r'\n', ' ', text)       # удаление переносов строк
    text = re.sub(r'[a-zA-Z]', ' ', text) # удаление английских символов
    text = re.sub(r'\s+', ' ', text)      # замена нескольких пробелов одним
    text = text.lower()                   # приведение к нижнему регистру
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        # Сохраняем во временный файл
        import tempfile
        import os
        
        # Создаем временную директорию, если её нет
        os.makedirs('/tmp', exist_ok=True)
        
        # Сохраняем во временный файл
        temp_path = f"/tmp/{pdf_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(pdf_file.getbuffer())
        
        st.info(f"Файл сохранен по пути: {temp_path}")
        
        # Открываем PDF из файла
        with fitz.open(temp_path) as doc:
            for page in doc:
                text += page.get_text()
        
        # Удаляем временный файл
        os.remove(temp_path)
        
    except Exception as e:
        st.error(f"Ошибка при чтении PDF: {e}")
        import traceback
        st.code(traceback.format_exc())
        
    return text

# Функция для загрузки модели
# Функция для загрузки модели
@st.cache_resource
def load_model():
    try:
        # Попытка загрузить существующую модель
        import joblib
        pipeline = joblib.load('resume_classifier.joblib')
        st.success("Модель успешно загружена из файла")
        return pipeline
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Функция для специфических комментариев в зависимости от класса
def get_class_specific_comment(class_id):
    # Предположим, что у нас классы 0 и 1, где 1 - положительное решение
    if class_id == 1:
        comments = [
            "Кандидат рекомендован. Профиль соответствует требованиям позиции.",
            "Резюме демонстрирует необходимые навыки и опыт для данной должности.",
            "Положительное решение. Квалификация кандидата соответствует ожиданиям."
        ]
    else:
        comments = [
            "Кандидат не рекомендован. Недостаточное соответствие требованиям.",
            "Резюме не содержит необходимого опыта для данной позиции.",
            "Профиль кандидата не соответствует критериям отбора."
        ]
    
    return random.choice(comments)

# Функция для извлечения ключевых слов из текста
def extract_key_words(text):
    # Простая эвристика для извлечения потенциальных ключевых слов
    # В реальном приложении здесь можно использовать более сложные методы,
    # например, TF-IDF или KeyBERT
    
    words = text.split()
    # Фильтруем короткие слова и оставляем только уникальные
    key_words = set([word for word in words if len(word) > 4])
    # Берем не более 20 слов
    return list(key_words)[:20]

# Функция для улучшения комментария на основе текста резюме
def enhance_comment_with_text(base_comment, text, predicted_class):
    # Если базовый комментарий пуст, создаем новый
    if not base_comment:
        base_comment = get_class_specific_comment(predicted_class)
    
    # Извлекаем текст резюме (до 1000 символов для анализа)
    text_sample = text[:1000].lower() if text else ""
    
    # Несколько шаблонов для дополнения базового комментария
    enhancements = []
    
    # Проверяем наличие опыта работы
    experience_indicators = ["опыт работы", "лет опыта", "год опыта", "опыт в", "работал в", "занимал должность"]
    for indicator in experience_indicators:
        if indicator in text_sample:
            enhancements.append("Имеет релевантный опыт работы.")
            break
    
    # Проверяем образование
    education_indicators = ["высшее образование", "университет", "вуз", "бакалавр", "магистр", "специальность"]
    for indicator in education_indicators:
        if indicator in text_sample:
            enhancements.append("Обладает подходящим образованием.")
            break
    
    # Проверяем технические навыки/компетенции
    skill_indicators = ["навыки", "владение", "знание", "умение", "компетенции", "технологии"]
    for indicator in skill_indicators:
        if indicator in text_sample:
            enhancements.append("Демонстрирует необходимые технические навыки.")
            break
    
    # Проверяем ключевые слова, связанные с продажами (из скриншота видно, что это важно)
    sales_indicators = ["продаж", "клиент", "менеджер", "сделк", "переговор", "презентаци"]
    for indicator in sales_indicators:
        if indicator in text_sample:
            enhancements.append("Имеет опыт в сфере продаж.")
            break
    
    # Дополняем базовый комментарий случайным образом, избегая повторений
    if enhancements:
        random.shuffle(enhancements)
        enhancements = enhancements[:2]  # Берем не более 2 дополнений
        enhanced_comment = base_comment + " " + " ".join(enhancements)
        return enhanced_comment
    
    return base_comment

# Функция для генерации комментария
def generate_comment(text, model, prediction):
    try:
        # Получаем базовый комментарий в зависимости от класса
        base_comment = get_class_specific_comment(prediction)
        
        # Улучшаем комментарий на основе текста резюме
        enhanced_comment = enhance_comment_with_text(base_comment, text, prediction)
        
        return enhanced_comment
    except Exception as e:
        st.warning(f"Ошибка при генерации комментария: {str(e)}")
        # В случае ошибки возвращаем простой комментарий
        if prediction == 1:
            return "Кандидат рекомендован к рассмотрению."
        else:
            return "Кандидат не рекомендован к рассмотрению."

# Главная функция приложения
def main():
    st.title("Классификация резюме с улучшенными комментариями")
    
    # Загрузка модели
    model = load_model()
    
    if model is None:
        st.error("Модель не загружена. Пожалуйста, убедитесь, что файл resume_classifier.joblib существует в директории.")
        st.stop()
    
    # Загрузка PDF файлов
    st.header("Загрузите PDF файлы для классификации")
    uploaded_files = st.file_uploader("Выберите PDF файлы", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Обрабатываем документы..."):
            results = []
            
            for file in uploaded_files:
                # Извлечение текста из PDF
                text = extract_text_from_pdf(file)
                clean_text_content = clean_text(text)
                
                if not clean_text_content:
                    st.warning(f"Не удалось извлечь текст из {file.name}")
                    continue
                
                try:
                    # Получение предсказания
                    prediction = model.predict([clean_text_content])[0]
                    # Получение вероятностей
                    proba = model.predict_proba([clean_text_content])[0]
                    
                    # Проверяем размерность proba
                    if len(proba) <= prediction:
                        st.warning(f"Ошибка: индекс {prediction} выходит за пределы размерности {len(proba)}")
                        relevance_prob = 0.5  # Значение по умолчанию
                    else:
                        relevance_prob = float(proba[prediction])
                    
                    # Генерация комментария
                    comment = generate_comment(text, model, prediction)
                    
                    # Добавление результата
                    results.append({
                        "file": file.name,
                        "predicted_class": int(prediction),
                        "relevance_prob": relevance_prob,
                        "comment": comment
                    })
                except Exception as e:
                    st.error(f"Ошибка при обработке файла {file.name}: {str(e)}")
            
            # Вывод результатов
            if results:
                st.success(f"Обработано {len(results)} файлов")
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Сохранение результатов
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()