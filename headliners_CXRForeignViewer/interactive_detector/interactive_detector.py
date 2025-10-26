import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from pathlib import Path
import pydicom as dicom

class DICOMAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM Image Annotator with Contrast Enhancement")
        self.root.geometry("1300x750")
        
        # Переменные
        self.current_image_path = None
        self.current_image_index = 0
        self.image_files = []
        self.processed_images = []  # Список ndarray
        self.original_images = []   # Оригинальные изображения без обработки
        self.image = None
        self.display_image = None
        self.photo = None
        self.rect_start = None
        self.rect_end = None
        self.current_rect = None
        self.annotations = []
        self.class_id = tk.StringVar(value="0")
        
        # Параметры для контрастирования
        self.method_var = tk.StringVar(value="simple")
        self.clip_limit_var = tk.StringVar(value="5.0")
        self.tile_size_var = tk.StringVar(value="8,8")
        self.patch_size_var = tk.StringVar(value="15")
        self.mode_var = tk.StringVar(value="gray")
        
        # Создание интерфейса
        self.create_widgets()
    
    def clahe_window(self, windowed, cL, tile, mode):
        """CLAHE контрастирование"""
        if mode == "gray":
            clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=tile)
            windowed = clahe.apply(windowed)
        else:
            if len(windowed.shape) == 2:
                windowed_rgb = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)
            else:
                windowed_rgb = windowed.copy()

            clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=tile)
            lab = cv2.cvtColor(windowed_rgb, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            windowed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            windowed = cv2.cvtColor(windowed, cv2.COLOR_BGR2GRAY)

        return windowed

    def dcp_window(self, windowed, patch_size):
        """Dark Channel Prior контрастирование"""
        if len(windowed.shape) == 2:
            windowed_rgb = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)
        else:
            windowed_rgb = windowed.copy()

        b, g, r = cv2.split(windowed_rgb)
        min_bg = cv2.min(b, g)
        dark_channel = cv2.min(min_bg, r)
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(dark_channel, kernel, iterations=1)

        return dark_channel

    def combined_window(self, windowed_rgb, cL, patch, tile, mode):
        """Комбинированный метод"""
        if mode == "rgb":
            clahed = self.clahe_window(windowed_rgb, cL, tile, mode)
        elif mode == "gray":
            clahed = self.clahe_window(windowed_rgb, cL, tile, mode)
            clahed = cv2.cvtColor(windowed_rgb, cv2.COLOR_GRAY2BGR)
        final = self.dcp_window(clahed, patch)
        return final

    def apply_window_level(self, image, window_center, window_width, photometric, method='simple', cL=5.0, tile=(8,8), patch=15, mode='gray'):
        """Применение оконного уровня и контрастирования"""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = image.copy()
        windowed[windowed < img_min] = img_min
        windowed[windowed > img_max] = img_max
        
        # Нормализация к 0-255 для отображения
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        if photometric == "MONOCHROME1":
            windowed = cv2.bitwise_not(windowed)

        # Применение методов контрастирования
        if method == 'clahe':
            windowed = self.clahe_window(windowed, cL, tile, mode)
        elif method == 'dcp':
            windowed = self.dcp_window(windowed, patch)
        elif method == 'combined':
            windowed = self.combined_window(windowed, cL, patch, tile, mode)
        
        return windowed

    def prepare_dicom_image(self, image_path: str, method='simple', cL=5.0, tile=(8,8), patch=5, mode="gray"):
        """Обработка DICOM файла с выбранным методом контрастирования"""
        # Читаем .dcm
        ds = dicom.dcmread(image_path)
        # Переводим в ndarray для обработки с помощью cv2
        dcm_sample = ds.pixel_array.astype(np.float32)

        center = getattr(ds, 'WindowCenter', None)
        if center is None:
            window_center = np.mean(ds.pixel_array)
        elif isinstance(center, dicom.multival.MultiValue):
            window_center = int(center[0]) if 'WindowCenter' in ds else np.mean(ds.pixel_array)
        else:
            window_center = int(center) if 'WindowCenter' in ds else np.mean(ds.pixel_array)

        width = getattr(ds, 'WindowWidth', None)
        if width is None:
            window_width = np.max(ds.pixel_array) - np.min(ds.pixel_array)
        elif isinstance(ds.WindowWidth, dicom.multival.MultiValue):
            window_width = int(ds.WindowWidth[0]) if 'WindowWidth' in ds else np.max(ds.pixel_array) - np.min(ds.pixel_array)
        else:
            window_width = int(ds.WindowWidth) if 'WindowWidth' in ds else np.max(ds.pixel_array) - np.min(ds.pixel_array)

        dcm_sample = self.apply_window_level(dcm_sample, window_center, window_width, 
                                           ds.PhotometricInterpretation, method, cL, tile, patch, mode)
        return dcm_sample

    def create_widgets(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - изображение
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Холст для изображения
        self.canvas = tk.Canvas(left_frame, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка событий мыши
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Правая панель - управление
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Путь к текущему файлу
        path_frame = ttk.LabelFrame(right_frame, text="Текущий файл", padding=5)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.path_label = ttk.Label(path_frame, text="Не выбран", wraplength=300)
        self.path_label.pack(fill=tk.X, pady=5)
        
        # Кнопки управления папкой
        folder_frame = ttk.LabelFrame(right_frame, text="Управление папкой", padding=5)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(folder_frame, text="Открыть папку с DICOM", 
                  command=self.open_folder).pack(fill=tk.X, pady=2)
        
        # Навигация по изображениям
        nav_frame = ttk.Frame(folder_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="← Пред", 
                  command=self.previous_image).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav_frame, text="След →", 
                  command=self.next_image).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.image_counter = ttk.Label(folder_frame, text="0/0")
        self.image_counter.pack(pady=2)
        
        # Настройки контрастирования
        contrast_frame = ttk.LabelFrame(right_frame, text="Настройки контрастирования", padding=10)
        contrast_frame.pack(fill=tk.X, pady=10)
        
        # Метод контрастирования
        ttk.Label(contrast_frame, text="Метод:").grid(row=0, column=0, sticky=tk.W, pady=2)
        method_combo = ttk.Combobox(contrast_frame, textvariable=self.method_var, 
                                   values=["simple", "clahe", "dcp", "combined"])
        method_combo.grid(row=0, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        method_combo.set("simple")
        
        # Clip Limit
        ttk.Label(contrast_frame, text="Clip Limit:").grid(row=1, column=0, sticky=tk.W, pady=2)
        clip_entry = ttk.Entry(contrast_frame, textvariable=self.clip_limit_var)
        clip_entry.grid(row=1, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # Tile Size
        ttk.Label(contrast_frame, text="Tile Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        tile_entry = ttk.Entry(contrast_frame, textvariable=self.tile_size_var)
        tile_entry.grid(row=2, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # Patch Size
        ttk.Label(contrast_frame, text="Patch Size:").grid(row=3, column=0, sticky=tk.W, pady=2)
        patch_entry = ttk.Entry(contrast_frame, textvariable=self.patch_size_var)
        patch_entry.grid(row=3, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # Mode
        ttk.Label(contrast_frame, text="Mode:").grid(row=4, column=0, sticky=tk.W, pady=2)
        mode_combo = ttk.Combobox(contrast_frame, textvariable=self.mode_var, 
                                 values=["gray", "rgb"])
        mode_combo.grid(row=4, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        mode_combo.set("gray")
        
        # Кнопка применения контрастирования
        apply_button = ttk.Button(contrast_frame, text="Применить контрастирование", 
                                 command=self.apply_contrast)
        apply_button.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))
        
        # Настройка весов колонок
        contrast_frame.columnconfigure(1, weight=1)
        
        # Кнопки управления разметкой
        control_frame = ttk.LabelFrame(right_frame, text="Управление разметкой", padding=5)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Сохранить разметку", 
                  command=self.save_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Очистить разметку", 
                  command=self.clear_annotations).pack(fill=tk.X, pady=2)
        
        # Ввод class_id
        class_frame = ttk.LabelFrame(right_frame, text="Class ID", padding=10)
        class_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(class_frame, text="Введите class_id:").pack(anchor=tk.W)
        class_entry = ttk.Entry(class_frame, textvariable=self.class_id)
        class_entry.pack(fill=tk.X, pady=5)
        
        # Область для отображения аннотаций
        annot_frame = ttk.LabelFrame(right_frame, text="Аннотации", padding=10)
        annot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Таблица аннотаций
        columns = ("class_id", "center_x", "center_y", "width", "height")
        self.tree = ttk.Treeview(annot_frame, columns=columns, show="headings", height=12)
        
        # Заголовки столбцов
        self.tree.heading("class_id", text="Class ID")
        self.tree.heading("center_x", text="Center X")
        self.tree.heading("center_y", text="Center Y")
        self.tree.heading("width", text="Width")
        self.tree.heading("height", text="Height")
        
        # Ширина столбцов
        self.tree.column("class_id", width=60)
        self.tree.column("center_x", width=70)
        self.tree.column("center_y", width=70)
        self.tree.column("width", width=70)
        self.tree.column("height", width=70)
        
        scrollbar = ttk.Scrollbar(annot_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Кнопка удаления выбранной аннотации
        ttk.Button(right_frame, text="Удалить выбранную", 
                  command=self.delete_selected).pack(fill=tk.X, pady=5)
        
        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def apply_contrast(self):
        """Применение выбранного метода контрастирования к текущему изображению"""
        if not self.original_images or self.current_image_index >= len(self.original_images):
            messagebox.showwarning("Предупреждение", "Нет изображения для обработки")
            return
        
        try:
            # Получаем параметры из полей ввода
            method = self.method_var.get()
            cL = float(self.clip_limit_var.get())
            tile = tuple(map(int, self.tile_size_var.get().split(',')))
            patch = int(self.patch_size_var.get())
            mode = self.mode_var.get()
            
            # Обрабатываем текущее изображение
            current_file = self.image_files[self.current_image_index]
            processed_image = self.prepare_dicom_image(current_file, method, cL, tile, patch, mode)
            
            # Обновляем обработанное изображение
            self.processed_images[self.current_image_index] = processed_image
            self.image = processed_image
            self.display_image = processed_image.copy()
            
            self.update_canvas()
            self.status_var.set(f"Применен метод: {method}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при применении контрастирования:\n{str(e)}")

    def open_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с DICOM файлами")
        
        if folder_path:
            self.load_dicom_folder(folder_path)

    def load_dicom_folder(self, folder_path):
        """Загрузка всех DICOM файлов из папки"""
        base_dir = Path(folder_path)
        self.image_files = []
        
        # Ищем все DICOM файлы
        for image_path in base_dir.rglob('*'):
            if image_path.is_file() and image_path.suffix.lower() in {'.dcm'}:
                self.image_files.append(str(image_path))
        
        if not self.image_files:
            messagebox.showwarning("Предупреждение", "В папке не найдены DICOM файлы")
            return
        
        # Обрабатываем все DICOM файлы (простая обработка по умолчанию)
        self.processed_images = []
        self.original_images = []
        
        for file_path in self.image_files:
            try:
                processed_image = self.prepare_dicom_image(file_path, method='simple')
                self.processed_images.append(processed_image)
                self.original_images.append(processed_image.copy())  # Сохраняем оригинал
            except Exception as e:
                print(f"Ошибка обработки {file_path}: {str(e)}")
                continue
        
        if not self.processed_images:
            messagebox.showerror("Ошибка", "Не удалось обработать ни одного DICOM файла")
            return
        
        # Показываем первое изображение
        self.current_image_index = 0
        self.show_current_image()
        
        self.status_var.set(f"Загружено {len(self.processed_images)} изображений")
        self.update_image_counter()

    def show_current_image(self):
        """Показывает текущее изображение"""
        if not self.processed_images or self.current_image_index >= len(self.processed_images):
            return
        
        self.image = self.processed_images[self.current_image_index]
        self.current_image_path = self.image_files[self.current_image_index]
        self.display_image = self.image.copy()
        
        # Обновляем путь к файлу
        self.path_label.config(text=os.path.basename(self.current_image_path))
        
        # Очищаем предыдущие аннотации
        self.clear_annotations()
        
        self.update_canvas()
        self.update_image_counter()
        
        # Показываем имя файла в статусе
        filename = os.path.basename(self.current_image_path)
        self.status_var.set(f"Текущий файл: {filename}")

    def update_image_counter(self):
        """Обновляет счетчик изображений"""
        total = len(self.processed_images)
        current = self.current_image_index + 1
        self.image_counter.config(text=f"{current}/{total}")

    def next_image(self):
        """Следующее изображение"""
        if self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def previous_image(self):
        """Предыдущее изображение"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def update_canvas(self):
        if self.display_image is None:
            return
            
        # Получаем размеры холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Если холст еще не отрисован, используем оригинальные размеры
            h, w = self.display_image.shape[:2]
            display_img = self.display_image
        else:
            # Масштабируем изображение под размер холста
            h, w = self.display_image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            display_img = cv2.resize(self.display_image, (new_w, new_h))
            self.scale_factor = scale
            self.canvas_offset_x = (canvas_width - new_w) // 2
            self.canvas_offset_y = (canvas_height - new_h) // 2
        
        # Конвертируем в формат для tkinter
        image_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Очищаем холст и отображаем изображение
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_offset_x, 
            self.canvas_offset_y, 
            anchor=tk.NW, 
            image=self.photo
        )
        
        # Перерисовываем аннотации
        self.redraw_annotations()

    def on_button_press(self, event):
        if self.image is None:
            return
            
        # Проверяем, что клик внутри изображения
        if (event.x < self.canvas_offset_x or event.y < self.canvas_offset_y):
            return
            
        self.rect_start = (event.x, event.y)
        self.current_rect = None

    def on_mouse_drag(self, event):
        if self.rect_start is None or self.image is None:
            return
            
        # Удаляем предыдущий прямоугольник
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Рисуем новый прямоугольник
        x1, y1 = self.rect_start
        x2, y2 = event.x, event.y
        
        # Ограничиваем прямоугольник областью изображения
        x1 = max(x1, self.canvas_offset_x)
        y1 = max(y1, self.canvas_offset_y)
        x2 = max(x2, self.canvas_offset_x)
        y2 = max(y2, self.canvas_offset_y)
        
        self.current_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="red", width=2
        )

    def on_button_release(self, event):
        if self.rect_start is None or self.image is None:
            return
            
        x1, y1 = self.rect_start
        x2, y2 = event.x, event.y
        
        # Проверяем, что выделение достаточно большое
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.rect_start = None
            if self.current_rect:
                self.canvas.delete(self.current_rect)
                self.current_rect = None
            return
        
        # Получаем координаты относительно изображения
        img_x1 = (x1 - self.canvas_offset_x) / self.scale_factor
        img_y1 = (y1 - self.canvas_offset_y) / self.scale_factor
        img_x2 = (x2 - self.canvas_offset_x) / self.scale_factor
        img_y2 = (y2 - self.canvas_offset_y) / self.scale_factor
        
        # Нормализуем координаты
        img_h, img_w = self.image.shape[:2]
        
        # Убеждаемся, что координаты в пределах изображения
        img_x1 = max(0, min(img_x1, img_w))
        img_y1 = max(0, min(img_y1, img_h))
        img_x2 = max(0, min(img_x2, img_w))
        img_y2 = max(0, min(img_y2, img_h))
        
        # Вычисляем центр, ширину и высоту
        center_x = (img_x1 + img_x2) / 2 / img_w
        center_y = (img_y1 + img_y2) / 2 / img_h
        width = abs(img_x2 - img_x1) / img_w
        height = abs(img_y2 - img_y1) / img_h
        
        # Получаем class_id
        try:
            class_id = int(self.class_id.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Class ID должен быть числом")
            return
        
        # Добавляем аннотацию
        annotation = {
            'class_id': class_id,
            'center_x': round(center_x, 6),
            'center_y': round(center_y, 6),
            'width': round(width, 6),
            'height': round(height, 6),
            'canvas_coords': (x1, y1, x2, y2)  # Для отрисовки
        }
        
        self.annotations.append(annotation)
        self.add_annotation_to_tree(annotation)
        
        # Рисуем постоянный прямоугольник
        self.draw_permanent_rectangle(annotation)
        
        self.rect_start = None
        self.current_rect = None
        
        self.status_var.set(f"Добавлена аннотация: class_id={class_id}")

    def draw_permanent_rectangle(self, annotation):
        x1, y1, x2, y2 = annotation['canvas_coords']
        annotation['rect_id'] = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="green", width=2
        )
        
        # Добавляем текст с class_id
        text_x = (x1 + x2) / 2
        text_y = y1 - 10
        annotation['text_id'] = self.canvas.create_text(
            text_x, text_y, text=str(annotation['class_id']), 
            fill="green", font=("Arial", 12, "bold")
        )

    def redraw_annotations(self):
        if not hasattr(self, 'scale_factor'):
            return
            
        for annotation in self.annotations:
            # Пересчитываем координаты для текущего масштаба
            img_h, img_w = self.image.shape[:2]
            
            # Восстанавливаем оригинальные координаты в пикселях
            center_x = annotation['center_x'] * img_w
            center_y = annotation['center_y'] * img_h
            width = annotation['width'] * img_w
            height = annotation['height'] * img_h
            
            # Вычисляем углы прямоугольника
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            # Конвертируем в координаты холста
            canvas_x1 = x1 * self.scale_factor + self.canvas_offset_x
            canvas_y1 = y1 * self.scale_factor + self.canvas_offset_y
            canvas_x2 = x2 * self.scale_factor + self.canvas_offset_x
            canvas_y2 = y2 * self.scale_factor + self.canvas_offset_y
            
            annotation['canvas_coords'] = (canvas_x1, canvas_y1, canvas_x2, canvas_y2)
            
            # Рисуем прямоугольник
            if 'rect_id' in annotation:
                self.canvas.delete(annotation['rect_id'])
                self.canvas.delete(annotation['text_id'])
            
            annotation['rect_id'] = self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2, 
                outline="green", width=2
            )
            
            # Текст с class_id
            text_x = (canvas_x1 + canvas_x2) / 2
            text_y = canvas_y1 - 10
            annotation['text_id'] = self.canvas.create_text(
                text_x, text_y, text=str(annotation['class_id']), 
                fill="green", font=("Arial", 12, "bold")
            )

    def add_annotation_to_tree(self, annotation):
        values = (
            annotation['class_id'],
            f"{annotation['center_x']:.6f}",
            f"{annotation['center_y']:.6f}",
            f"{annotation['width']:.6f}",
            f"{annotation['height']:.6f}"
        )
        self.tree.insert("", tk.END, values=values)

    def delete_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            return
            
        # Находим индекс выбранного элемента
        index = self.tree.index(selected_item[0])
        
        # Удаляем с холста
        annotation = self.annotations[index]
        self.canvas.delete(annotation['rect_id'])
        self.canvas.delete(annotation['text_id'])
        
        # Удаляем из списка и из дерева
        self.annotations.pop(index)
        self.tree.delete(selected_item[0])
        
        self.status_var.set("Аннотация удалена")

    def clear_annotations(self):
        # Удаляем все аннотации с холста
        for annotation in self.annotations:
            if 'rect_id' in annotation:
                self.canvas.delete(annotation['rect_id'])
            if 'text_id' in annotation:
                self.canvas.delete(annotation['text_id'])
        
        # Очищаем списки
        self.annotations.clear()
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Все аннотации очищены")

    def save_annotations(self):
        if not self.current_image_path or not self.annotations:
            messagebox.showwarning("Предупреждение", "Нет аннотаций для сохранения")
            return
        
        # Создаем имя файла для аннотаций (тот же путь, но с расширением .txt)
        base_name = os.path.splitext(self.current_image_path)[0]
        annot_file = base_name + ".txt"
        
        try:
            with open(annot_file, 'w') as f:
                for annotation in self.annotations:
                    line = (f"{annotation['class_id']} "
                           f"{annotation['center_x']:.6f} "
                           f"{annotation['center_y']:.6f} "
                           f"{annotation['width']:.6f} "
                           f"{annotation['height']:.6f}\n")
                    f.write(line)
            
            self.status_var.set(f"Аннотации сохранены в: {os.path.basename(annot_file)}")
            messagebox.showinfo("Успех", f"Аннотации сохранены в:\n{annot_file}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def on_resize(self, event):
        if self.image is not None:
            self.update_canvas()

def main():
    root = tk.Tk()
    app = DICOMAnnotator(root)
    
    # Обработка изменения размера окна
    root.bind("<Configure>", app.on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()