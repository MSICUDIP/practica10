from auxiliar import *
from procedimientos import *

import os
import sys

import cv2

def main():

    # Asegura el directorio de trabajo para respetar rutas relativas
    if len(os.path.dirname(__file__)) > 0 :
        os.chdir(os.path.dirname(__file__))
   
    # Carga una imagen RGB
    imagen = cargar_imagen_rgb("../resources/pintora_de_luna_urueta_1952.png")
    imagen_elefante = cargar_imagen_rgb("../resources/circo_maria_izquierdo.png")
    
    # Separa los planos de color de la imagen
    R,G,B = separar_planos_color(imagen)
    
    # Crear imagen BGR
    # BGR es el modelo estandar de OpenCV.
    #
    # Entonces hay que guardar la imagen en BGR para
    # desplegarla correctamente con OpenCV.
    #
    # Otras bibliotecas usan RGB por default.
    imagen_bgr = integrar_planos_color(B, G, R)
    mostrar_imagen(imagen_bgr, "Original")
    
    mostrar_imagen(R, "R")
    guardar_imagen(R, "../resources/R.png")
    mostrar_imagen(G, "G")
    guardar_imagen(G, "../resources/G.png")
    mostrar_imagen(B, "B")
    guardar_imagen(B, "../resources/B.png")
    
    # Calcular imagen de color inversa
    imagen_ymc = integrar_planos_color(255 - B,
                                       255 - G,
                                       255 - R)
    mostrar_imagen(imagen_ymc, "Imagen YMC")
    guardar_imagen(imagen_ymc, "../resources/YMC.png")
    
    # Convierte a HSI
    H, S, I = convertir_a_hsi(imagen)

    # Normaliza a 255
    H = normalizar_255(H)
    S = normalizar_255(S)
    I = normalizar_255(I)

    mostrar_imagen(H, "H")
    guardar_imagen(H, "../resources/H.png")
    mostrar_imagen(S, "S")
    guardar_imagen(S, "../resources/S.png")
    mostrar_imagen(I, "I")
    guardar_imagen(I, "../resources/I.png")
   
    # Integrar los planos de HSI en una sola imagen
    imagen_hsi = integrar_planos_color(H, S, I)
    mostrar_imagen(imagen_hsi, "Imagen HSI")
    guardar_imagen(imagen_hsi, "../resources/HSI.png")
   
    # Convertir a BGR la imagen del elefante
    R,G,B = separar_planos_color(imagen_elefante)
    imagen_bgr = integrar_planos_color(B,G,R)
    mostrar_imagen(imagen_bgr, "Original (imagen elefante)")

    # Convertir a HSI la imagen del elefante
    H,S,I = convertir_a_hsi(imagen_elefante)
    mostrar_imagen(normalizar_255(H), "H (imagen elefante)")
    mostrar_imagen(normalizar_255(S), "S (imagen elefante)")
    mostrar_imagen(normalizar_255(I), "I (imagen elefante)")

    # Segmentar en H
    mascara = segmentar(H)
    mascara_255 = mascara * 255
    imagen_elefante = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara)
    np.savez_compressed("../solutions/imagen_elefante", solucion=imagen_elefante)
    mostrar_imagen(imagen_elefante, "Imagen Segmentada")
    guardar_imagen(imagen_elefante, "../resources/elefante_segmentado.png")

if __name__ == "__main__":
    main()
