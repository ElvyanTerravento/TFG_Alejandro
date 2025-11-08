package tfgalejandro;

import javax.swing.text.JTextComponent;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Puente sencillo para exponer métodos compatibles con Swing que PanelTraduccion
 * invoca por reflexión si están disponibles. Puede envolver un ControllerUI real
 * o funcionar de manera autónoma para las operaciones básicas.
 */
public class ControllerUISwingBridge {

    public ControllerUISwingBridge() {}

    /**
     * Borra el último carácter del área de texto (seguro ante cadenas vacías).
     */
    public void onTranslationDeleteLastSwing(JTextComponent area) {
        if (area == null) return;
        String txt = area.getText();
        if (txt != null && !txt.isEmpty()) {
            area.setText(txt.substring(0, txt.length() - 1));
        }
    }

    /**
     * Limpia completamente el área de texto.
     */
    public void onTranslationClearAllSwing(JTextComponent area) {
        if (area == null) return;
        area.setText("");
    }
    
    // -------------------- Predicción desde imagen (bridge) --------------------
    private static final String PY_SCRIPT_DIR = "C:\\Users\\alex1\\Desktop\\APP_JESUS_CASI_FINIT\\JESUS_APP\\TFG_Alejandro\\resources_local";
    private static final String PYTHON_EXE = System.getenv().getOrDefault("GESTURE_PYTHON_EXE", "python");
    // Script local para inferencia clásica (dentro del proyecto)
    private static final String LOCAL_INFER_SCRIPT = new java.io.File("resources_local/infer_local_svm.py").getAbsolutePath();
    // Script local para inferencia DL (Keras)
    private static final String DL_INFER_SCRIPT = new java.io.File("resources_local/infer_dl.py").getAbsolutePath();

    /**
     * Realiza una predicción a partir de una imagen capturada.
     * - Guarda la imagen como PNG temporal
     * - Invoca los métodos privados de ControllerUI por reflexión si hay instancia
     * - Si no hay instancia o falla, intenta vía Python local para Gemini/OpenAI
     * Devuelve el token normalizado (letra, SPACE, DEL, etc.) o cadena vacía si no hay resultado.
     */
    public String predictFromImage(BufferedImage image, String selectedModel) {
        if (image == null) return "";
        String model = selectedModel != null ? selectedModel.trim() : "";
        Path tmp = null;
        try {
            tmp = Files.createTempFile("sign_frame_", ".png");
            ImageIO.write(image, "PNG", tmp.toFile());
            String imagePath = tmp.toAbsolutePath().toString();

            // Ejecutar script Python local según modelo (LLMs) o dejar vacío para clásicos no soportados aún
            String raw;
            String lower = model.toLowerCase(Locale.ROOT);
            if (lower.startsWith("gemini-")) {
                raw = runGeminiSingleLocal(imagePath, model);
            } else if (lower.startsWith("gpt-")) {
                raw = runOpenAISingleLocal(imagePath, model);
            } else if (lower.startsWith("dl_file=")) {
                String f = model.substring("dl_file=".length());
                raw = runDlModelPredict(imagePath, f);
            } else if (lower.startsWith("local_dir=")) {
                // Ruta absoluta a carpeta de modelo clásico
                String dir = model.substring("local_dir=".length());
                raw = runLocalClassicPredict(imagePath, dir);
            } else {
                raw = runLocalClassicPredict(imagePath, model);
            }

            // Extraer overlay si lo hubiera
            String overlay = extractOverlayPath(raw);
            // Filtrar y devolver solo el valor de "Resultado: ..." -> token permitido
            String candidate = extractResultadoToken(raw);
            String token = normalizeToAllowedToken(raw, candidate);
            if (token == null) token = "";
            if (overlay != null && !overlay.isBlank()) {
                return token + "||OVERLAY:" + overlay;
            }
            return token;
        } catch (Exception e) {
            return "";
        } finally {
            if (tmp != null) {
                try { Files.deleteIfExists(tmp); } catch (IOException ignore) {}
            }
        }
    }

    
    private String runGeminiSingleLocal(String imagePath, String model) {
        if (model == null || model.isBlank()) return "";
        String baseModel = model.replaceAll("\\s*\\(.*\\)$", "").trim();
        if (!baseModel.startsWith("gemini-")) return "";
        String scriptPath = new java.io.File(PY_SCRIPT_DIR, "vision_GEMINI.py").getAbsolutePath();
        System.out.println("[API CALL][GEMINI] model=" + baseModel + ", image=" + imagePath + ", script=" + scriptPath);
        StringBuilder code = new StringBuilder();
        code.append("import importlib.util, sys\n");
        code.append("_p = r'" + scriptPath.replace("\\", "\\\\") + "'\n");
        code.append("_spec = importlib.util.spec_from_file_location('vision_gemini_mod', _p)\n");
        code.append("_mod = importlib.util.module_from_spec(_spec)\n");
        code.append("_spec.loader.exec_module(_mod)\n");
        code.append("_mod.SINGLE_IMAGE_PATH = r'" + imagePath.replace("\\", "\\\\") + "'\n");
        code.append("_mod.BATCH_DIR = None\n");
        code.append("_mod.MODEL_NAME = '" + baseModel + "'\n");
        code.append("_client = _mod.GeminiClient(_mod.MODEL_NAME)\n");
        code.append("_mod.run_single(_client)\n");
        String out = execPythonLocal(code.toString());
        System.out.println("[API RESPONSE][GEMINI] " + out);
        return out;
    }

    private String runOpenAISingleLocal(String imagePath, String model) {
        if (model == null || model.isBlank()) return "";
        String baseModel = model.replaceAll("\\s*\\(.*\\)$", "").trim();
        if (!baseModel.startsWith("gpt-")) return "";
        String scriptPath = new java.io.File(PY_SCRIPT_DIR, "vision_OPENAI.py").getAbsolutePath();
        System.out.println("[API CALL][OPENAI] model=" + baseModel + ", image=" + imagePath + ", script=" + scriptPath);
        StringBuilder code = new StringBuilder();
        code.append("import importlib.util, sys\n");
        code.append("_p = r'" + scriptPath.replace("\\", "\\\\") + "'\n");
        code.append("_spec = importlib.util.spec_from_file_location('vision_openai_mod', _p)\n");
        code.append("_mod = importlib.util.module_from_spec(_spec)\n");
        code.append("_spec.loader.exec_module(_mod)\n");
        code.append("_mod.SINGLE_IMAGE_PATH = r'" + imagePath.replace("\\", "\\\\") + "'\n");
        code.append("_mod.BATCH_DIR = None\n");
        code.append("_mod.MODEL_NAME = '" + baseModel + "'\n");
        code.append("from openai import OpenAI\n");
        code.append("try:\n    _api = _mod.load_api_key()\n    _client = OpenAI(api_key=_api)\nexcept Exception:\n    _client = OpenAI()\n");
        code.append("_mod.run_single(_client)\n");
        String out = execPythonLocal(code.toString());
        System.out.println("[API RESPONSE][OPENAI] " + out);
        return out;
    }

    private String runDlModelPredict(String imagePath, String modelFile) {
        if (modelFile == null || modelFile.isBlank()) return "";
        System.out.println("[DL INFERENCE] Ejecutando script: " + DL_INFER_SCRIPT + " con modelo: " + modelFile);
        ProcessBuilder pb = new ProcessBuilder(
                PYTHON_EXE,
                DL_INFER_SCRIPT,
                "--image", imagePath,
                "--model-file", modelFile
        );
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream(), java.nio.charset.StandardCharsets.UTF_8))) {
                StringBuilder out = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println("[DL INFERENCE][PY] " + line);
                    out.append(line).append(System.lineSeparator());
                }
                p.waitFor();
                return out.toString().trim();
            }
        } catch (Exception e) {
            System.out.println("[DL INFERENCE] Error ejecutando script: " + e.getMessage());
            return "";
        }
    }

    private String runLocalClassicPredict(String imagePath, String modelDir) {
        if (modelDir == null || modelDir.isBlank()) return "";
        File dir = new File(modelDir);
        if (!dir.isDirectory()) {
            System.out.println("[LOCAL INFERENCE] La ruta proporcionada no es una carpeta válida: " + modelDir);
            return "";
        }
        // Llamar al script Python local
        System.out.println("[LOCAL INFERENCE] Ejecutando script: " + LOCAL_INFER_SCRIPT);
        ProcessBuilder pb = new ProcessBuilder(
                PYTHON_EXE,
                LOCAL_INFER_SCRIPT,
                "--image", imagePath,
                "--model-dir", dir.getAbsolutePath()
        );
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream(), java.nio.charset.StandardCharsets.UTF_8))) {
                StringBuilder out = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println("[LOCAL INFERENCE][PY] " + line);
                    out.append(line).append(System.lineSeparator());
                }
                p.waitFor();
                return out.toString().trim();
            }
        } catch (Exception e) {
            System.out.println("[LOCAL INFERENCE] Error ejecutando script: " + e.getMessage());
            return "";
        }
    }

    // Extrae el contenido después de "Resultado:" (case-insensitive). Si no existe, intenta devolver la última línea no vacía.
    private String extractResultadoToken(String raw) {
        if (raw == null) return null;
        String[] lines = raw.split("\r?\n");
        String lastNonEmpty = null;
        for (String ln : lines) {
            if (ln == null) continue;
            String trimmed = ln.trim();
            if (!trimmed.isEmpty()) lastNonEmpty = trimmed;
            String lower = trimmed.toLowerCase(Locale.ROOT);
            if (lower.contains("resultado")) {
                int idx = trimmed.indexOf(':');
                String val = (idx >= 0 && idx + 1 < trimmed.length()) ? trimmed.substring(idx + 1).trim() : trimmed;
                return val;
            }
        }
        return lastNonEmpty; // Fallback: última línea útil
    }

    // Normaliza a los tokens permitidos: una letra A-Z, o DEL / NOTHING / SPACE
    private String normalizeToAllowedToken(String fullOutput, String candidate) {
        // 1) Intentar detectar palabras clave en candidate
        String t = tryKeywords(candidate);
        if (t != null) return t;
        // 2) Si no, detectar primera letra A-Z en candidate
        String letter = firstAZ(candidate);
        if (letter != null) return letter;
        // 3) Fallback: buscar en toda la salida por keywords y luego por letra
        t = tryKeywords(fullOutput);
        if (t != null) return t;
        return firstAZ(fullOutput);
    }

    private String tryKeywords(String s) {
        if (s == null) return null;
        String up = s.toUpperCase(Locale.ROOT);
        if (up.contains("DEL")) return "DEL";
        if (up.contains("NOTHING")) return "NOTHING";
        if (up.contains("SPACE")) return "SPACE";
        return null;
    }

    private String firstAZ(String s) {
        if (s == null) return null;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'A' && c <= 'Z') return String.valueOf(c);
            if (c >= 'a' && c <= 'z') return String.valueOf(Character.toUpperCase(c));
        }
        return null;
    }

    private String extractOverlayPath(String raw) {
        if (raw == null) return null;
        String[] lines = raw.split("\r?\n");
        for (String ln : lines) {
            if (ln == null) continue;
            String t = ln.trim();
            String up = t.toUpperCase(Locale.ROOT);
            if (up.startsWith("OVERLAY:")) {
                String val = t.substring(t.indexOf(':') + 1).trim();
                return val;
            }
        }
        return null;
    }

    private String execPythonLocal(String code) {
        ProcessBuilder pb = new ProcessBuilder(PYTHON_EXE, "-c", code);
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder out = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    out.append(line).append(System.lineSeparator());
                }
                p.waitFor();
                return out.toString().trim();
            }
        } catch (Exception e) {
            return "";
        }
    }

}
