(ns clj-predict-pth.core
  (:require [clojure.data.json :as json])
  (:import
   ;;[ai.djl Engine]
   [ai.djl.modality.cv.transform Resize ToTensor Normalize]
   [ai.djl.modality.cv.translator ImageClassificationTranslator]
   [ai.djl.translate Translator Batchifier]
   [ai.djl.modality.cv Image ImageFactory]
   [ai.djl.modality Classifications]
   [ai.djl.util.cuda CudaUtils]
   [ai.djl.repository.zoo Criteria]
   [ai.djl.ndarray NDManager NDArray]
   [ai.djl Device]
   [java.nio.file Paths]))

(defn get-device []
  (if (zero? (CudaUtils/getGpuCount))
    (Device/cpu)
    (Device/gpu)))

(def class-names
  (-> (slurp "/home/jc/Work/BirdStriker/REPO/class_names.json")
      (json/read-str)))

(def bird-classfier nil)

(defn get-translator [transform-image-size]
  (-> (ImageClassificationTranslator/builder)
      (.addTransform (Resize. transform-image-size transform-image-size))
      (.addTransform (ToTensor.))
      ;; Assuming ImageNet normalization values - adjust as needed
      (.addTransform (Normalize. (float-array [0.485 0.456 0.406])
                                 (float-array [0.229 0.224 0.225])))
      (.build)))

(defn get-predictor [model-pathname]
  (or bird-classfier
      (let [criteria (-> (Criteria/builder)
                         (.setTypes Image Classifications)
                         (.optEngine "PyTorch")
                         (.optDevice (get-device))
                         (.optModelPath (Paths/get model-pathname (into-array String [])))
                         (.optTranslator (get-translator 384))
                         (.build))
            predictor (.. criteria loadModel newPredictor)]
        (alter-var-root #'bird-classfier (constantly predictor))
        bird-classfier)))

(defn predict-top-5-birds [model-pathname image-pathname]
  (let [image (->> []
                   (into-array String)
                   (Paths/get image-pathname)
                   (.fromFile (ImageFactory/getInstance)))
        predictor (get-predictor model-pathname)]
    (-> (.predict predictor image)
        str
        json/read-str)))

(comment
  (spit "/home/jc/Work/BirdStriker/REPO/synset.txt"
        (clojure.string/join "\n" (map second (sort-by (comp Integer/parseInt first) class-names))))
  (predict-top-5-birds "/home/jc/Work/BirdStriker/REPO/model_complete.pt" "/home/jc/Work/BirdStriker/REPO/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
  (predict-top-5-birds "/home/jc/Work/BirdStriker/REPO/model_complete.pt" "/home/jc/Work/BirdStriker/REPO/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
  )
