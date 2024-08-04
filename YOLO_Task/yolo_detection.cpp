// Ce code est écrit par BigVision LLC. Il est basé sur le projet OpenCV. 
// Il est soumis aux termes de la licence dans le fichier LICENSE trouvé dans cette distribution et sur http://opencv.org/license.html

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{help h usage ? | | Exemples d'utilisation : \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| image d'entrée   }"
"{video v       |<none>| vidéo d'entrée   }"
"{device d       |<cpu>| appareil d'entrée   }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialiser les paramètres
float confThreshold = 0.5; // Seuil de confiance
float nmsThreshold = 0.4;  // Seuil de suppression non-maximale
int inpWidth = 416;  // Largeur de l'image d'entrée du réseau
int inpHeight = 416; // Hauteur de l'image d'entrée du réseau
vector<string> classes;

// Supprimer les boîtes englobantes avec une faible confiance en utilisant la suppression non-maximale
void postprocess(Mat& frame, const vector<Mat>& out);

// Dessiner la boîte englobante prédite
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Obtenir les noms des couches de sortie
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Utilisez ce script pour exécuter la détection d'objets en utilisant YOLO3 dans OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Charger les noms des classes
    string classesFile = "classes.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = "cpu";
    device = parser.get<String>("device");
    
    // Donner les fichiers de configuration et de poids pour le modèle
    String modelConfiguration = "yolov3_custom.cfg";
    String modelWeights = "yolov3-custom_last.weights";

    // Charger le réseau
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device == "cpu")
    {
        cout << "Utilisation de l'appareil CPU" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
        cout << "Utilisation de l'appareil GPU" << endl;
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }

    // Ouvrir un fichier vidéo ou une image
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Ouvrir le fichier image
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("erreur");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Ouvrir le fichier vidéo
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("erreur");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        else
        {
            throw("Aucun fichier image ou vidéo spécifié");
        }
        
    }
    catch(...) {
        cout << "Impossible d'ouvrir le flux d'image/vidéo d'entrée" << endl;
        return 0;
    }
    
    // Initialiser le writer vidéo pour enregistrer la vidéo de sortie
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    
    // Créer une fenêtre
    static const string kWinName = "Détection d'objets par apprentissage profond dans OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Traiter les frames
    while (waitKey(1) < 0)
    {
        // obtenir une frame de la vidéo
        cap >> frame;

        // Arrêter le programme si la fin de la vidéo est atteinte
        if (frame.empty()) {
            cout << "Traitement terminé !!!" << endl;
            cout << "Le fichier de sortie est enregistré sous " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Créer un blob 4D à partir d'une frame
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        // Définir l'entrée au réseau
        net.setInput(blob);
        
        // Exécuter la passe avant pour obtenir la sortie des couches de sortie
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Supprimer les boîtes englobantes avec une faible confiance
        postprocess(frame, outs);
        
        // Mettre les informations d'efficacité. La fonction getPerfProfile retourne le temps global pour l'inférence(t) et les temps pour chacune des couches(dans layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Temps d'inférence pour une frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Écrire la frame avec les boîtes de détection
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image")) imwrite(outputFile, detectedFrame);
        else video.write(detectedFrame);
        
        imshow(kWinName, frame);
        
    }
    
    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}

// Supprimer les boîtes englobantes avec une faible confiance en utilisant la suppression non-maximale
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Parcourir toutes les boîtes englobantes de sortie du réseau et ne garder que celles avec des scores de confiance élevés. Attribuer l'étiquette de classe de la boîte comme la classe avec le score le plus élevé pour la boîte.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Obtenir la valeur et l'emplacement du score maximum
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Effectuer la suppression non maximale pour éliminer les boîtes redondantes se chevauchant avec des confiances plus faibles
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}
// Dessiner la boîte englobante prédite
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    // Dessiner un rectangle affichant la boîte englobante
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    // Obtenir l'étiquette pour le nom de la classe et sa confiance
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    // Afficher l'étiquette en haut de la boîte englobante
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 1);
}

// Obtenir les noms des couches de sortie
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Obtenir les indices des couches de sortie, c'est-à-dire les couches avec des sorties non connectées
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        // Obtenir les noms de toutes les couches dans le réseau
        vector<String> layersNames = net.getLayerNames();
        
        // Obtenir les noms des couches de sortie dans names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
