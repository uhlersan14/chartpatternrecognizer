package ch.zhaw.chartvision.chartvision.service;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ChartPatternRecognizer {
    private static final Logger logger = LoggerFactory.getLogger(ChartPatternRecognizer.class);
    private Predictor<Image, Classifications> predictor;

    public ChartPatternRecognizer() {
        try {
            // Prüfe, ob das trainierte Modell existiert
            Path modelPath = Paths.get("./trained-models/chart-patterns");
            Path synsetPath = Paths.get("./trained-models/synset.txt");
            
            if (Files.exists(modelPath) && Files.exists(synsetPath)) {
                logger.info("Loading custom trained model...");
                // Verwende dein trainiertes Modell
                Criteria<Image, Classifications> criteria = Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optOption("synsetPath", synsetPath.toString())
                        .optDevice(Device.cpu())
                        .optProgress(new ProgressBar())
                        .build();
                
                ZooModel<Image, Classifications> model = criteria.loadModel();
                predictor = model.newPredictor();
            } else {
                logger.warn("No trained model found. Using pretrained ResNet for demo purposes.");
                // Verwende ein vortrainiertes Modell als Fallback
                Criteria<Image, Classifications> criteria = Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optFilter("backbone", "resnet50") // Verwende ResNet50
                        .optFilter("dataset", "imagenet")  // ImageNet dataset
                        .optDevice(Device.cpu())
                        .optProgress(new ProgressBar())
                        .build();
                
                ZooModel<Image, Classifications> model = criteria.loadModel();
                predictor = model.newPredictor();
            }
            
            logger.info("Chart pattern recognition model loaded successfully");
            
        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            logger.error("Failed to load model", e);
            throw new RuntimeException("Could not initialize chart pattern recognizer", e);
        }
    }
    
    public Classifications predict(Image image) throws Exception {
        if (predictor == null) {
            throw new IllegalStateException("Predictor not initialized");
        }
        return predictor.predict(image);
    }
}