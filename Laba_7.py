# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:19:01 2024

@author: Vanya
"""

import cv2  
import numpy as np
import matplotlib.pyplot as plt
import imutils

# 1 Поиск шаблона на изображении

# Загружаем изображение
rgb_img = cv2.imread('image.jpg') 
plt.figure()
plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

# Преобразуем изображение в оттенки серого 
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
  
# Загружаем шаблон  
template = cv2.imread('template.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))

# Поворот на 90 градусов
rotated_image = imutils.rotate(template, angle=90)

# В цикле будем увеличивать масштаб картинки
layer = gray_img.copy() 
for i in range(3):
    # Функция pyrDown() уменьшает масштаб изображения, а  pyrUp() - увеличивает
    # По умолчанию масштаб изменяется в два раза
    layer = cv2.pyrUp(layer)
    # cv2.imshow("str(i)", layer)
    # cv2.waitKey(0)

    # Преобразуем в оттенки серого
    gray_templ = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)  
    
    
    # Считаем размеры шаблона
    w, h = rotated_image.shape[:-1] 
     
    # Вызываем функцию cv2.matchTemplate для вычисления метрики схожести
    # в качестве параметров передаем изображение, шаблон и тип вычисляемой метрики
    res = cv2.matchTemplate(gray_img,gray_templ,cv2.TM_CCOEFF_NORMED)  
    
    # Возможные варианты метрик:
    #    cv2.TM_SQDIFF — сумма квадратов разниц значений пикселей
    #    cv2.TM_SQDIFF_NORMED — сумма квадрат разниц цветов, отнормированная в диапазон 0..1.
    #    cv2.TM_CCORR — сумма поэлементных произведений шаблона и сегмента картинки
    #    cv2.TM_CCORR_NORMED — сумма поэлементных произведений, отнормированное в диапазон -1..1.
    #    cv2.TM_CCOEFF — кросс-коррелация изображений без среднего
    #    cv2.TM_CCOEFF_NORMED — кросс-корреляция между изображениями без среднего, отнормированная в -1..1 (корреляция Пирсона)
    plt.figure()
    plt.imshow(res, cmap='jet')
    plt.colorbar()
    
    
    # Определяем порог для выделения области локализации шаблона на изобажении
    # Порог зависит от метрики, т.к. значения различных метрик могут различаться
    # на порядки. Кроме по своей сути некоторые метрики измеряют "похожесть" 
    # и имеют большие значения для похожих изображений, а другие измеряют "отличие",
    # и ноборот, большие значения появляются для различающихся изображений
    threshold = 0.7
    
    # Определяем точки изображения в которых метрика превышает порог
    # Эти точки - центры локализации шаблона
    # Знак сравнения для метрик, измеряющих "отличия" необходимо заменить на противоположный
    loc = np.where(res >= threshold)  
    
    # Вокруг выделенных максимумов обводим прямоугольники с размерами шаблона
    plot_img = rgb_img.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(plot_img, pt,(pt[0] + w, pt[1] + h),(0,255,255), 8)  
    
    # Отображаем результат на графике
    plt.figure()
    plt.imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))