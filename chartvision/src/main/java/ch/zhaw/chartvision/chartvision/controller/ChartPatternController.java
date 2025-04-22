package ch.zhaw.chartvision.chartvision.controller;

import ch.zhaw.chartvision.chartvision.service.ChartPatternRecognizer;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.awt.image.BufferedImage;
import java.io.IOException;
import javax.imageio.ImageIO;

@RestController
@RequestMapping("/api/chart")
public class ChartPatternController {
    
    private final ChartPatternRecognizer patternRecognizer = new ChartPatternRecognizer();
    
    @GetMapping("/ping")
    public String ping() {
        return "ChartVision app is up and running!";
    }
    
    @PostMapping("/analyze")
    public String analyzeChart(@RequestParam("image") MultipartFile file) {
        try {
            // Konvertiere die hochgeladene Datei in ein DJL Image
            BufferedImage bufferedImage = ImageIO.read(file.getInputStream());
            Image djlImage = ImageFactory.getInstance().fromImage(bufferedImage);
            
            // Führe die Prediction durch
            Classifications result = patternRecognizer.predict(djlImage);
            
            // Gib das Ergebnis als JSON zurück
            return result.toJson();
            
        } catch (IOException e) {
            return "{\"error\": \"Failed to process image: " + e.getMessage() + "\"}";
        } catch (Exception e) {
            return "{\"error\": \"Analysis failed: " + e.getMessage() + "\"}";
        }
    }
}