package ch.zhaw.chartvision.chartvision.training;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.listener.TrainingListener;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Repository;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.Blocks;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;

public class ChartPatternTraining {
    
    public static void trainModel(String datasetPath, String modelSavePath) throws Exception {
        // Dataset laden
        Repository repository = Repository.newInstance("image_folder", Paths.get(datasetPath));
        ImageFolder dataset = ImageFolder.builder()
                .setRepository(repository)
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .setSampling(32, true)  // batch size 32
                .build();
        
        // Vorbereitung des Datasets
        dataset.prepare();
        
        // Dataset in Training und Validation aufteilen
        RandomAccessDataset[] splits = dataset.randomSplit(8, 2); // 80% training, 20% validation
        RandomAccessDataset trainingSet = splits[0];
        RandomAccessDataset validationSet = splits[1];
        
        // Modell erstellen
        Model model = Model.newInstance("chart-pattern-model");
        
        // Einfaches CNN-Modell definieren
        Block block = createCnnBlock(dataset.getSynset().size());
        model.setBlock(block);
        
        // Training-Konfiguration
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());
        
        // Trainer erstellen
        try (Trainer trainer = model.newTrainer(config)) {
            // Input Shape setzen
            Shape inputShape = new Shape(1, 3, 224, 224); // batch, channels, height, width
            trainer.initialize(inputShape);
            
            // Training durchführen
            int epochs = 10;
            EasyTrain.fit(trainer, epochs, trainingSet, validationSet);
            
            // Modell speichern
            Path modelDir = Paths.get(modelSavePath);
            model.save(modelDir, "chart-patterns");
            
            // Synset (Klassennamen) speichern
            try (java.io.PrintWriter writer = new java.io.PrintWriter(modelDir.resolve("synset.txt").toFile())) {
                for (String className : dataset.getSynset()) {
                    writer.println(className);
                }
            }
            
            System.out.println("Model saved successfully to: " + modelSavePath);
        }
    }
    
    private static Block createCnnBlock(int numClasses) {
        SequentialBlock block = new SequentialBlock();
        
        // Layer 1: Conv -> BatchNorm -> ReLU -> MaxPool
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optStride(new Shape(1, 1))
                .optPadding(new Shape(1, 1))
                .setFilters(32)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        
        // Layer 2: Conv -> BatchNorm -> ReLU -> MaxPool
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optStride(new Shape(1, 1))
                .optPadding(new Shape(1, 1))
                .setFilters(64)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        
        // Flatten und Fully Connected Layer
        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(numClasses).build());
        
        return block;
    }
    
    public static void main(String[] args) {
        try {
            // Pfade anpassen
            String datasetPath = "./kaggle-crypto-charts";
            String modelSavePath = "./trained-models";
            
            trainModel(datasetPath, modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}