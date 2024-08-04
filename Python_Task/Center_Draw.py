import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
from rembg import remove

def adjust_saturation(image, increase_saturation=True, decrease_channels=None):
    """
    Ajuste la saturation de l'image.
    - `increase_saturation` si True, augmente la saturation ; sinon, ne fait rien.
    - `decrease_channels` liste des canaux pour diminuer la saturation ('G', 'B')
    """
    if len(image.shape) == 3:
        # Convertir l'image en HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        if increase_saturation:
            s = cv2.add(s, 50)  # Augmente la saturation en ajoutant 50 au canal de saturation

        if decrease_channels:
            if 'G' in decrease_channels:
                s = s * 0.5  # Diminuer la saturation pour le canal vert
            if 'B' in decrease_channels:
                s = s * 0.5  # Diminuer la saturation pour le canal bleu
        
        # Clipper les valeurs pour qu'elles soient dans la plage valide [0, 255]
        s = np.clip(s, 0, 255).astype(np.uint8)

        # Fusionner les canaux et convertir de nouveau en BGR
        hsv_image = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return image

def remove_background(input_image):
    # Supprimer l'arrière-plan en utilisant rembg
    return remove(input_image)

def canny_edge_detector(image, low_threshold, high_threshold):
    # Convertir l'image en niveaux de gris si elle est en couleur
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Appliquer un lissage gaussien
    blurred_image = cv2.GaussianBlur(image_gray, (1, 1), 0)

    # Appliquer le détecteur de contours de Canny
    edge_map = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edge_map

def process_images(image_folder_path, low_threshold, high_threshold, output_pdf_path, output_png_folder):
    # Assurer que le dossier de sortie existe
    os.makedirs(output_png_folder, exist_ok=True)

    # Obtenir la liste de tous les chemins des fichiers d'images dans le dossier
    image_paths = glob.glob(os.path.join(image_folder_path, '*'))

    if not image_paths:
        raise FileNotFoundError(f"Aucune image trouvée dans le dossier '{image_folder_path}'. Veuillez vérifier le chemin du dossier.")

    print(f"{len(image_paths)} images trouvées dans le dossier '{image_folder_path}'.")

    # Créer un objet PdfPages pour sauvegarder les figures
    with PdfPages(output_pdf_path) as pdf:
        for idx, image_path in enumerate(image_paths):
            try:
                print(f"Traitement de l'image : {image_path}")

                # Lire l'image originale
                original_image = cv2.imread(image_path)
                
                # Détecter les contours dans l'image originale
                original_edge_map = canny_edge_detector(original_image, low_threshold, high_threshold)

                # Ajuster la saturation
                adjusted_image = adjust_saturation(original_image, increase_saturation=True, decrease_channels=['G', 'B'])
                
                # Supprimer l'arrière-plan
                _, buffer = cv2.imencode('.png', adjusted_image)
                output_image = remove_background(buffer.tobytes())
                output_image = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
                
                # Détecter les contours dans l'image avec l'arrière-plan supprimé
                edge_map_with_bg_removed = canny_edge_detector(output_image, low_threshold, high_threshold)
                
                # Calculer le centre de gravité dans la carte des contours de l'image avec l'arrière-plan supprimé
                moments = cv2.moments(edge_map_with_bg_removed)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = edge_map_with_bg_removed.shape[1] // 2, edge_map_with_bg_removed.shape[0] // 2  # Centre de secours
                
                # Afficher les coordonnées du centre de gravité
                print(f"Centre de gravité pour l'image {idx + 1} : (x: {cx}, y: {cy})")

                # Dessiner le centre de gravité sur l'image originale
                original_image_with_point = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                cv2.circle(original_image_with_point, (cx, cy), 5, (255, 0, 0), -1)

                # Dessiner le centre de gravité sur l'image avec l'arrière-plan supprimé
                output_image_with_point = cv2.cvtColor(output_image, cv2.COLOR_BGRA2RGB)
                cv2.circle(output_image_with_point, (cx, cy), 5, (255, 0, 0), -1)

                # Créer une nouvelle figure pour chaque image
                plt.figure(figsize=(20, 10), dpi=200)

                # Image originale avec point
                plt.subplot(2, 2, 1)
                plt.imshow(original_image_with_point)
                plt.title(f'Image Originale {idx + 1}')
                plt.axis('off')

                # Détection des contours sur l'image originale
                plt.subplot(2, 2, 2)
                plt.imshow(original_edge_map, cmap='gray')
                plt.title(f'Contours Image Originale {idx + 1}')
                plt.axis('off')

                # Image avec arrière-plan supprimé et point
                plt.subplot(2, 2, 3)
                plt.imshow(output_image_with_point)
                plt.title(f'Image sans arrière-plan {idx + 1}')
                plt.axis('off')

                # Détection des contours sur l'image avec l'arrière-plan supprimé
                plt.subplot(2, 2, 4)
                plt.imshow(edge_map_with_bg_removed, cmap='gray')
                plt.title(f'Contours sans arrière-plan {idx + 1}')
                plt.axis('off')

                # Ajuster la disposition pour un meilleur espacement
                plt.subplots_adjust(wspace=0.3, hspace=0.5)

                # Sauvegarder la figure actuelle dans le PDF
                pdf.savefig()

                # Sauvegarder la figure actuelle en tant que fichier PNG
                png_filename = os.path.join(output_png_folder, f'processed_image_{idx + 1}.png')
                plt.savefig(png_filename)
                plt.close()

            except FileNotFoundError as e:
                print(e)

    print(f"PDF sauvegardé à {output_pdf_path}")
    print(f"Fichiers PNG sauvegardés dans le dossier '{output_png_folder}'")

if __name__ == "__main__":
    # Obtenir le répertoire actuel du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Définir le chemin relatif vers le dossier d'images
    image_folder_path = os.path.join(script_dir, 'DataPart1')  
    low_threshold = 50
    high_threshold = 150
    output_pdf_path = os.path.join(script_dir, 'images_pré-traitées.pdf')  # Chemin où le PDF sera sauvegardé
    output_png_folder = os.path.join(script_dir, 'images_pré-traitées')  # Dossier où les fichiers PNG seront sauvegardés
    
    process_images(image_folder_path, low_threshold, high_threshold, output_pdf_path, output_png_folder)
