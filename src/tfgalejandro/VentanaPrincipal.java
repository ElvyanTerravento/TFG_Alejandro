/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package tfgalejandro;

import java.awt.Cursor;
import java.awt.image.BufferedImage;
import tfgalejandro.ControllerUISwingBridge;
import javax.swing.SwingUtilities;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import javax.swing.DefaultComboBoxModel;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Paths;

/**
 *
 * @author jesus
 */
public class VentanaPrincipal extends javax.swing.JFrame {

    private ControllerUISwingBridge controllerBridge;
    // Cola y control de orden de inferencias
    private final ExecutorService inferencePool = Executors.newFixedThreadPool(4);
    private final AtomicLong seqGen = new AtomicLong(1);
    private final AtomicLong nextToFlush = new AtomicLong(1);
    private final ConcurrentHashMap<Long, String> pendingResults = new ConcurrentHashMap<>();
    private final AtomicLong totalQueued = new AtomicLong(0);
    private final AtomicLong processed = new AtomicLong(0);
    // Mapa de modelos locales (label -> ruta absoluta)
    private final Map<String, String> localModelMap = new HashMap<>();
    // Lista de etiquetas locales detectadas (sin separadores)
    private final List<String> localLabels = new ArrayList<>();
    // Modelos DL (archivo -> ruta absoluta)
    private final Map<String, String> dlModelMap = new HashMap<>();
    private final List<String> dlLabels = new ArrayList<>();
    // Cache maestro de modelos API (constante) + locales (sin separadores)
    private List<String> allApiModels = java.util.Arrays.asList(API_MODELS);

    // Listas permitidas por tipo de clasificador DL
    private static final String[] DL_CNN_ALLOWED = new String[] {
        "cnn_ASL.h5",
        "cnn_MNIST.h5",
        "resnet50_ASL_DOBLE.h5",
        "resnet50_ASL.h5",
        "resnet50_MNIST_DOBLE.h5",
        "resnet50_MNIST.h5"
    };
    private static final String DL_MLP_ALLOWED = "MLP_ASL.keras";
    private static final String DL_GCN_ALLOWED = "GCN_ASL.keras";

    /**
     * Creates new form VentanaPrincipal
     */
    public VentanaPrincipal() {
        initComponents();
        setExtendedState(javax.swing.JFrame.MAXIMIZED_BOTH);

        this.panelCompletar.setRandomWord();

        listaClasificadores.addItem("Otro clasificador");

    // Conectar PanelTraduccion con un bridge compatible con Swing.
        this.controllerBridge = new ControllerUISwingBridge();
        this.panelTraduccionApp.setController(this.controllerBridge);

        // Población de modelos API (Gemini / OpenAI) y ajuste de combos
        initApiModelCombos();
        // Añadir modelos locales (clásicos) y modelos DL desde dl_models
        populateLocalModelCombos();
        populateDlModels();
        // Inicializar filtros descriptor/clasificador y aplicar
        initFilterCombos();
        applyModelFilters();
    }

    // Modelos API disponibles (ajusta esta lista si añades más modelos)
    private static final String[] API_MODELS = new String[] {
        "gemini-2.5-flash (simple)",
        "gemini-2.5-pro (simple)",
        "gemini-2.5-flash (complex)",
        "gemini-2.5-pro (complex)",
        "gpt-4o (simple)",
        "gpt-4o (complex)"
    };

    private void initApiModelCombos() {
        // Sustituir contenido del combo de modelos por los modelos API
        javax.swing.DefaultComboBoxModel<String> model = new javax.swing.DefaultComboBoxModel<>(API_MODELS);
        this.listaModelos.setModel(model);
        if (model.getSize() > 0) {
            this.listaModelos.setSelectedIndex(0);
        }
        // Preparar combos de descriptor y clasificador (filtros)
    // Descriptores: añadir también CNN/MLP/GCN para modelos DL
    this.listaDescriptores.setModel(new javax.swing.DefaultComboBoxModel<>(
        new String[] {"Todos", "HOG", "LBP", "HOG+LBP", "CNN", "MLP", "GCN", "API"}));
    // Clasificadores: SVMs + familias DL y API
    this.listaClasificadores.setModel(new javax.swing.DefaultComboBoxModel<>(
        new String[] {"Todos", "SVM 0.1", "SVM 0.01", "CNN", "MLP", "GCN", "API"}));
        this.listaDescriptores.setEnabled(true);
        this.listaClasificadores.setEnabled(true);
        this.listaDescriptores.setToolTipText("Filtra por descriptor del modelo (HOG, LBP, combinados o API)");
        this.listaClasificadores.setToolTipText("Filtra por clasificador del modelo (SVM 0.1 / 0.01 o API)");
        this.listaModelos.setToolTipText("Selecciona un modelo para inferencia (API o LOCAL)");

        // Fijar ancho visual estable de los combos con elementos "prototipo"
        this.listaModelos.setPrototypeDisplayValue("LOCAL: HOG_MNIST+LBP_MNIST_MNIST_SVM_C0p01");
    this.listaDescriptores.setPrototypeDisplayValue("HOG+LBP");
    this.listaClasificadores.setPrototypeDisplayValue("ResNet50 (CNN)");
        // Evitar que el toolbar estire los combos más allá de su tamaño preferido
        java.awt.Dimension dm = this.listaModelos.getPreferredSize();
        this.listaModelos.setMaximumSize(dm);
        java.awt.Dimension dd = this.listaDescriptores.getPreferredSize();
        this.listaDescriptores.setMaximumSize(dd);
        java.awt.Dimension dc = this.listaClasificadores.getPreferredSize();
        this.listaClasificadores.setMaximumSize(dc);
    }

    private void initFilterCombos() {
        try {
            this.listaDescriptores.addActionListener(e -> applyModelFilters());
            this.listaClasificadores.addActionListener(e -> applyModelFilters());
        } catch (Exception ignore) {}
    }

    private void applyModelFilters() {
        try {
            String descFilter = (String) this.listaDescriptores.getSelectedItem();
            String clfFilter = (String) this.listaClasificadores.getSelectedItem();
            if (descFilter == null) descFilter = "Todos";
            if (clfFilter == null) clfFilter = "Todos";

            DefaultComboBoxModel<String> filtered = new DefaultComboBoxModel<>();

            // 1) API models
            for (String api : API_MODELS) {
                if (matchesFilters(api, true, descFilter, clfFilter)) {
                    filtered.addElement(api);
                }
            }

            // 2) Local models (clásicos)
            List<String> localsMatched = new ArrayList<>();
            for (String label : localLabels) {
                if (matchesFilters(label, false, descFilter, clfFilter)) {
                    localsMatched.add(label);
                }
            }
            if (!localsMatched.isEmpty()) {
                filtered.addElement("──────── LOCAL MODELS ────────");
                for (String l : localsMatched) filtered.addElement(l);
            }

            // 3) DL models
            List<String> dlMatched = new ArrayList<>();
            for (String label : dlLabels) {
                if (matchesFilters(label, false, descFilter, clfFilter)) {
                    dlMatched.add(label);
                }
            }
            if (!dlMatched.isEmpty()) {
                filtered.addElement("──────── DL MODELS ────────");
                for (String l : dlMatched) filtered.addElement(l);
            }

            String prevSel = (String) this.listaModelos.getSelectedItem();
            this.listaModelos.setModel(filtered);
            // Restaurar selección si sigue presente
            if (prevSel != null) {
                for (int i = 0; i < filtered.getSize(); i++) {
                    if (prevSel.equals(filtered.getElementAt(i))) { this.listaModelos.setSelectedIndex(i); return; }
                }
            }
            if (filtered.getSize() > 0) this.listaModelos.setSelectedIndex(0);
        } catch (Exception ex) {
            System.out.println("[FILTER] Error aplicando filtros: " + ex.getMessage());
        }
    }

    private boolean matchesFilters(String label, boolean isApi, String descFilter, String clfFilter) {
        // Normalizar filtros
        boolean anyDesc = descFilter == null || descFilter.equals("Todos");
        boolean anyClf = clfFilter == null || clfFilter.equals("Todos");
        if (isApi) {
            String d = "API";
            String c = "API";
            boolean okDesc = anyDesc || d.equalsIgnoreCase(descFilter);
            boolean okClf = anyClf || c.equalsIgnoreCase(clfFilter);
            return okDesc && okClf;
        }
        // Local/DL: derivar d/c desde el nombre base tras "LOCAL: " o "DL: "
        String base = label;
        if (base.startsWith("LOCAL: ")) base = base.substring(7);
        if (base.startsWith("DL: ")) base = base.substring(4);
        String up = base.toUpperCase();
        String d;
        if (up.endsWith(".H5") || up.contains("RESNET")) d = "CNN"; // Keras H5 / ResNet50
        else if (up.endsWith(".KERAS") && up.startsWith("MLP_")) d = "MLP";
        else if (up.endsWith(".KERAS") && up.startsWith("GCN_")) d = "GCN";
        else if (up.startsWith("HOG_") && up.contains("+LBP_")) d = "HOG+LBP";
        else if (up.startsWith("HOG_")) d = "HOG";
        else if (up.startsWith("LBP_")) d = "LBP";
        else d = "";
        String c;
        if ("CNN".equals(d)) c = "CNN";
        else if ("MLP".equals(d)) c = "MLP";
        else if ("GCN".equals(d)) c = "GCN";
        else if (up.contains("SVM_C0P1")) c = "SVM 0.1";
        else if (up.contains("SVM_C0P01")) c = "SVM 0.01";
        else c = "";
        boolean okDesc = anyDesc || d.equalsIgnoreCase(descFilter);
        boolean okClf = anyClf || c.equalsIgnoreCase(clfFilter);
        return okDesc && okClf;
    }

    // Escanea la carpeta de modelos locales y añade entradas al combo
    private void populateLocalModelCombos() {
        try {
            // Ruta base de modelos: calcular relativa al directorio de trabajo del proceso
            String basePath = Paths.get(System.getProperty("user.dir"), "models").toString();
            File baseDir = new File(basePath);
            if (!baseDir.exists() || !baseDir.isDirectory()) {
                baseDir = new File("models");
            }
            if (!baseDir.exists() || !baseDir.isDirectory()) {
                System.out.println("[LOCAL MODELS] Carpeta de modelos no encontrada: " + baseDir.getAbsolutePath());
                return;
            }

            System.out.println("[LOCAL MODELS] Escaneando base: " + baseDir.getAbsolutePath());

            File[] subdirs = baseDir.listFiles(File::isDirectory);
            if (subdirs == null || subdirs.length == 0) {
                System.out.println("[LOCAL MODELS] No hay subcarpetas de modelos en: " + baseDir.getAbsolutePath());
                return;
            }

            DefaultComboBoxModel<String> model = (DefaultComboBoxModel<String>) this.listaModelos.getModel();
            boolean addedHeader = false;

            int found = 0;
            for (File d : subdirs) {
                // 1) Comprobar directamente la carpeta de primer nivel
                String lbl = detectValidModelLabel(d);
                if (lbl != null) {
                    if (!addedHeader) { model.addElement("──────── LOCAL MODELS ────────"); addedHeader = true; }
                    model.addElement(lbl); found++;
                    localLabels.add(lbl);
                }

                // 2) Si tiene subcarpeta "models" u otras, escanear un nivel más por si los ficheros están dentro
                File[] level2 = d.listFiles(File::isDirectory);
                if (level2 != null) {
                    for (File d2 : level2) {
                        String lbl2 = detectValidModelLabel(d2);
                        if (lbl2 != null) {
                            if (!addedHeader) { model.addElement("──────── LOCAL MODELS ────────"); addedHeader = true; }
                            model.addElement(lbl2); found++;
                            localLabels.add(lbl2);
                        }
                    }
                }
            }
            System.out.println("[LOCAL MODELS] Total modelos locales añadidos: " + found);
        } catch (Exception e) {
            System.out.println("[LOCAL MODELS] Error al poblar modelos locales: " + e.getMessage());
        }
    }

    // Escanea carpeta dl_models para modelos DL y añade entradas al combo
    private void populateDlModels() {
        try {
            // Ruta base por defecto: ./dl_models (fallback a la carpeta dl_models dentro del working dir)
            File baseDir = new File("dl_models");
            if (!baseDir.exists() || !baseDir.isDirectory()) {
                baseDir = new File(Paths.get(System.getProperty("user.dir"), "dl_models").toString());
            }
            if (!baseDir.exists() || !baseDir.isDirectory()) {
                System.out.println("[DL MODELS] Carpeta dl_models no encontrada: " + baseDir.getAbsolutePath());
                return;
            }

            System.out.println("[DL MODELS] Escaneando base: " + baseDir.getAbsolutePath());

            File[] files = baseDir.listFiles();
            if (files == null || files.length == 0) return;

            DefaultComboBoxModel<String> model = (DefaultComboBoxModel<String>) this.listaModelos.getModel();
            boolean addedHeader = false;
            int found = 0;

            for (File f : files) {
                if (f.isFile()) {
                    String name = f.getName();
                    String up = name.toUpperCase();
                    boolean isCnn = up.endsWith(".H5");
                    boolean isMlp = up.endsWith(".KERAS") && up.startsWith("MLP_");
                    boolean isGcn = up.endsWith(".KERAS") && up.startsWith("GCN_");

                    // Filtrado según reglas del usuario
                    boolean allowed = false;
                    if (isCnn) {
                        for (String n : DL_CNN_ALLOWED) { if (n.equals(name)) { allowed = true; break; } }
                    } else if (isMlp) {
                        allowed = DL_MLP_ALLOWED.equals(name);
                    } else if (isGcn) {
                        allowed = DL_GCN_ALLOWED.equals(name);
                    }

                    if (allowed) {
                        String label = "DL: " + name;
                        if (!dlModelMap.containsKey(label)) {
                            dlModelMap.put(label, f.getAbsolutePath());
                            dlLabels.add(label);
                            if (!addedHeader) { model.addElement("──────── DL MODELS ────────"); addedHeader = true; }
                            model.addElement(label);
                            found++;
                            System.out.println("[DL MODELS] Detectado: " + f.getAbsolutePath());
                        }
                    }
                }
            }
            System.out.println("[DL MODELS] Total modelos DL añadidos: " + found);
        } catch (Exception e) {
            System.out.println("[DL MODELS] Error al poblar modelos DL: " + e.getMessage());
        }
    }

    private String detectValidModelLabel(File dir) {
        try {
            if (dir == null || !dir.isDirectory()) return null;
            File coef = new File(dir, "coef.npy");
            File inter = new File(dir, "intercept.npy");
            File scaler = new File(dir, "scaler_params.npz");
            File[] joblibs = dir.listFiles(f -> f.isFile() && f.getName().toLowerCase().endsWith(".joblib"));
            boolean looksValid = (coef.exists() && inter.exists()) || (joblibs != null && joblibs.length > 0);
            if (!looksValid) return null;

            // Si la carpeta válida se llama genéricamente "models", usa el nombre del padre para el label
            String displayName;
            if ("models".equalsIgnoreCase(dir.getName()) && dir.getParentFile() != null) {
                displayName = dir.getParentFile().getName();
            } else {
                displayName = dir.getName();
            }
            String label = "LOCAL: " + displayName;
            // Evitar duplicados si mismo label ya se añadió
            if (localModelMap.containsKey(label)) return null;
            localModelMap.put(label, dir.getAbsolutePath());

            System.out.println("[LOCAL MODELS] Detectado: " + dir.getAbsolutePath());
            if (joblibs != null) for (File jl : joblibs) System.out.println("  - " + jl.getName());
            if (coef.exists()) System.out.println("  - coef.npy");
            if (inter.exists()) System.out.println("  - intercept.npy");
            if (scaler.exists()) System.out.println("  - scaler_params.npz");
            return label;
        } catch (Exception ex) {
            System.out.println("[LOCAL MODELS] Error analizando dir: " + dir + " -> " + ex.getMessage());
            return null;
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPopupMenu1 = new javax.swing.JPopupMenu();
        Modelo1 = new javax.swing.JRadioButtonMenuItem();
        splitPanePrincipal = new javax.swing.JSplitPane();
        panelTabuladoAplicaciones = new javax.swing.JTabbedPane();
        panelTraduccionApp = new tfgalejandro.PanelTraduccion();
        panelCompletarApp = new javax.swing.JPanel();
        panelCompletar = new tfgalejandro.PanelCompletar();
        panelAutenticacionApp = new javax.swing.JPanel();
        panelAutenticacion = new tfgalejandro.PanelAutenticacion();
        splitPaneCamara = new javax.swing.JSplitPane();
        panelCamara = new tfgalejandro.PanelCamara();
        panelImagen = new tfgalejandro.PanelImagen();
        barraHerramientas = new javax.swing.JToolBar();
        botonCamara = new javax.swing.JToggleButton();
        botonCaptura = new javax.swing.JButton();
        listaModelos = new javax.swing.JComboBox<>();
        listaDescriptores = new javax.swing.JComboBox<>();
        listaClasificadores = new javax.swing.JComboBox<>();
        barraEstado = new javax.swing.JLabel();
        barraMenu = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        jMenuItem1 = new javax.swing.JMenuItem();

        Modelo1.setSelected(true);
        Modelo1.setText("jRadioButtonMenuItem1");
        jPopupMenu1.add(Modelo1);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Sign reader");
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                formWindowClosing(evt);
            }
        });

        splitPanePrincipal.setDividerLocation(500);

        panelTabuladoAplicaciones.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.LOWERED));
        panelTabuladoAplicaciones.addTab("Traducción", panelTraduccionApp);

        panelCompletarApp.setLayout(new java.awt.BorderLayout());
        panelCompletarApp.add(panelCompletar, java.awt.BorderLayout.CENTER);

        panelTabuladoAplicaciones.addTab("Completa palabra", panelCompletarApp);

        panelAutenticacionApp.setLayout(new java.awt.BorderLayout());
        panelAutenticacionApp.add(panelAutenticacion, java.awt.BorderLayout.CENTER);

        panelTabuladoAplicaciones.addTab("Autenticación", panelAutenticacionApp);

        splitPanePrincipal.setRightComponent(panelTabuladoAplicaciones);

        splitPaneCamara.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.LOWERED));
        splitPaneCamara.setDividerLocation(200);
        splitPaneCamara.setOrientation(javax.swing.JSplitPane.VERTICAL_SPLIT);
        splitPaneCamara.setLeftComponent(panelCamara);

        javax.swing.GroupLayout panelImagenLayout = new javax.swing.GroupLayout(panelImagen);
        panelImagen.setLayout(panelImagenLayout);
        panelImagenLayout.setHorizontalGroup(
            panelImagenLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 496, Short.MAX_VALUE)
        );
        panelImagenLayout.setVerticalGroup(
            panelImagenLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 391, Short.MAX_VALUE)
        );

        splitPaneCamara.setRightComponent(panelImagen);

        splitPanePrincipal.setLeftComponent(splitPaneCamara);

        getContentPane().add(splitPanePrincipal, java.awt.BorderLayout.CENTER);

        barraHerramientas.setRollover(true);

        botonCamara.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/Camara.png"))); // NOI18N
        botonCamara.setSelected(true);
        botonCamara.setFocusable(false);
        botonCamara.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        botonCamara.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        botonCamara.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                botonCamaraActionPerformed(evt);
            }
        });
        barraHerramientas.add(botonCamara);

        botonCaptura.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/Capturar.png"))); // NOI18N
        botonCaptura.setFocusable(false);
        botonCaptura.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        botonCaptura.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        botonCaptura.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                botonCapturaActionPerformed(evt);
            }
        });
        barraHerramientas.add(botonCaptura);

    labelModelo = new javax.swing.JLabel();
    labelModelo.setText("Modelo:");
    labelModelo.setBorder(javax.swing.BorderFactory.createEmptyBorder(0, 8, 0, 4));
    barraHerramientas.add(labelModelo);

        listaModelos.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "HOG_SL_SVM", "Modelo 2" }));
        barraHerramientas.add(listaModelos);

    labelDescriptor = new javax.swing.JLabel();
    labelDescriptor.setText("Descriptor:");
    labelDescriptor.setBorder(javax.swing.BorderFactory.createEmptyBorder(0, 8, 0, 4));
    barraHerramientas.add(labelDescriptor);

        listaDescriptores.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "LBP", "Descriptor 2" }));
        barraHerramientas.add(listaDescriptores);

    labelClasificador = new javax.swing.JLabel();
    labelClasificador.setText("Clasificador:");
    labelClasificador.setBorder(javax.swing.BorderFactory.createEmptyBorder(0, 8, 0, 4));
    barraHerramientas.add(labelClasificador);

        listaClasificadores.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "SVM", "Clasificador 2" }));
        barraHerramientas.add(listaClasificadores);

        getContentPane().add(barraHerramientas, java.awt.BorderLayout.NORTH);

        barraEstado.setText("Estado");
        getContentPane().add(barraEstado, java.awt.BorderLayout.PAGE_END);

        jMenu1.setText("Archivo");

        jMenuItem1.setText("Abrir");
        jMenu1.add(jMenuItem1);

        barraMenu.add(jMenu1);

        setJMenuBar(barraMenu);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void botonCapturaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_botonCapturaActionPerformed
        if (this.panelCamara.getCamera() != null) {               
            BufferedImage imagen = this.panelCamara.getCamera().getImage();            
            // Mostrar inmediatamente el frame capturado
            this.panelImagen.setImagen(imagen);
            this.repaint();

            // Preparar datos de inferencia
            String selected = (String)this.listaModelos.getSelectedItem();
            // Si el modelo es API y no hay conexión, forzar fallback a DL ResNet50_ASL_DOBLE y reflejarlo en el combo
            if (selected != null && isApiModel(selected)) {
                boolean online = hasInternetConnection();
                if (!online) {
                    String fallback = selectFallbackResNetModelLabel();
                    if (fallback != null) {
                        ensureModelInComboAndSelect(fallback);
                        selected = fallback;
                        try { barraEstado.setText("Sin conexión: usando modelo local " + fallback); } catch (Exception ignore) {}
                    }
                }
            }
            String modelo;
            if (selected != null && localModelMap.containsKey(selected)) {
                // Pasar la ruta absoluta incrustada en el parámetro del modelo
                String dir = localModelMap.get(selected);
                modelo = "LOCAL_DIR=" + dir;
            } else if (selected != null && dlModelMap.containsKey(selected)) {
                String file = dlModelMap.get(selected);
                modelo = "DL_FILE=" + file;
            } else {
                modelo = selected;
            }

            // Encolar tarea con número de secuencia para respetar el orden de escritura
            long seq = seqGen.getAndIncrement();
            totalQueued.incrementAndGet();
            updateEstadoLabel();

            inferencePool.submit(() -> {
                String output = (this.controllerBridge != null)
                        ? this.controllerBridge.predictFromImage(imagen, modelo)
                        : "";
                if (output == null) output = "";

                // Procesar overlay si viene adjunto (formato TOKEN||OVERLAY:<path>)
                String tokenOut = output;
                String overlayPath = null;
                int sep = output.indexOf("||OVERLAY:");
                if (sep >= 0) {
                    tokenOut = output.substring(0, sep);
                    overlayPath = output.substring(sep + "||OVERLAY:".length());
                }
                final String overlayFinal = overlayPath;
                if (overlayFinal != null && !overlayFinal.isBlank()) {
                    try {
                        java.awt.image.BufferedImage over = ImageIO.read(new java.io.File(overlayFinal));
                        if (over != null) {
                            SwingUtilities.invokeLater(() -> {
                                this.panelImagen.setImagen(over);
                                this.panelImagen.repaint();
                            });
                        }
                    } catch (Exception ex) {
                        System.out.println("[DL OVERLAY] No se pudo cargar overlay: " + ex.getMessage());
                    }
                }

                pendingResults.put(seq, tokenOut);
                flushResultsInOrder();
            });
        }
    }//GEN-LAST:event_botonCapturaActionPerformed

    private void formWindowClosing(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosing
        this.panelCamara.close();
        try { inferencePool.shutdownNow(); } catch (Exception ignore) {}
    }//GEN-LAST:event_formWindowClosing

    private void botonCamaraActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_botonCamaraActionPerformed
        if(panelCamara.getCamera()!=null){
           this.panelCamara.close(); 
           this.pack();
           this.repaint();
        } else {
            setCursor(new Cursor(Cursor.WAIT_CURSOR));
            this.panelCamara.startWebCam();
            this.pack();
            this.repaint();
            setCursor(Cursor.getDefaultCursor());
        }
    }//GEN-LAST:event_botonCamaraActionPerformed



    

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JRadioButtonMenuItem Modelo1;
    private javax.swing.JLabel barraEstado;
    private javax.swing.JToolBar barraHerramientas;
    private javax.swing.JMenuBar barraMenu;
    private javax.swing.JToggleButton botonCamara;
    private javax.swing.JButton botonCaptura;
    private javax.swing.JLabel labelClasificador;
    private javax.swing.JLabel labelDescriptor;
    private javax.swing.JLabel labelModelo;
    private javax.swing.JMenu jMenu1;
    private javax.swing.JMenuItem jMenuItem1;
    private javax.swing.JPopupMenu jPopupMenu1;
    private javax.swing.JComboBox<String> listaClasificadores;
    private javax.swing.JComboBox<String> listaDescriptores;
    private javax.swing.JComboBox<String> listaModelos;
    private tfgalejandro.PanelAutenticacion panelAutenticacion;
    private javax.swing.JPanel panelAutenticacionApp;
    private tfgalejandro.PanelCamara panelCamara;
    private tfgalejandro.PanelCompletar panelCompletar;
    private javax.swing.JPanel panelCompletarApp;
    private tfgalejandro.PanelImagen panelImagen;
    private javax.swing.JTabbedPane panelTabuladoAplicaciones;
    private tfgalejandro.PanelTraduccion panelTraduccionApp;
    private javax.swing.JSplitPane splitPaneCamara;
    private javax.swing.JSplitPane splitPanePrincipal;
    // End of variables declaration//GEN-END:variables

    // ---------- Helpers de cola/estado ----------
    private void updateEstadoLabel() {
        SwingUtilities.invokeLater(() -> {
            long t = totalQueued.get();
            long p = processed.get();
            if (t > 0 || p > 0) {
                barraEstado.setText("Estado: En cola " + p + "/" + t);
                // indicar busy mientras hay trabajos en cola
                setCursor(new Cursor(Cursor.WAIT_CURSOR));
            } else {
                barraEstado.setText("Estado: Listo");
                setCursor(Cursor.getDefaultCursor());
            }
        });
    }

    private void flushResultsInOrder() {
        SwingUtilities.invokeLater(() -> {
            while (true) {
                long next = nextToFlush.get();
                String res = pendingResults.remove(next);
                if (res == null) break;
                nextToFlush.incrementAndGet();
                processed.incrementAndGet();
                // Escribir resultado en la pestaña activa
                switch(panelTabuladoAplicaciones.getSelectedIndex()){
                    case 0 -> { if (res != null) this.panelTraduccionApp.addTexto(res); }
                    case 1 -> this.panelCompletar.addTextoSigno(res);
                    case 2 -> this.panelAutenticacion.addTextoClave(res);
                }
            }
            // Actualizar estado y reset si hemos terminado lote
            long t = totalQueued.get();
            long p = processed.get();
            if (p >= t && t > 0) {
                totalQueued.set(0);
                processed.set(0);
                // show ready
                barraEstado.setText("Estado: Listo");
                setCursor(Cursor.getDefaultCursor());
            } else {
                barraEstado.setText("Estado: En cola " + p + "/" + t);
                setCursor(new Cursor(Cursor.WAIT_CURSOR));
            }
        });
    }

    // ---------- Conectividad y fallback de modelo ----------
    private boolean isApiModel(String label) {
        if (label == null) return false;
        for (String api : API_MODELS) {
            if (label.equals(api)) return true;
        }
        return false;
    }

    private boolean hasInternetConnection() {
        // Intenta URLs conocidas con timeout corto para no bloquear el EDT
        String[] probes = new String[] {
            "https://clients3.google.com/generate_204",
            "https://www.google.com/generate_204",
            "https://www.gstatic.com/generate_204"
        };
        for (String u : probes) {
            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(u).openConnection();
                conn.setConnectTimeout(1500);
                conn.setReadTimeout(1500);
                conn.setInstanceFollowRedirects(false);
                conn.setRequestMethod("GET");
                int code = conn.getResponseCode();
                if (code == 204 || code == 200) return true;
            } catch (Exception ignore) { /* try next */ }
        }
        return false;
    }

    private String selectFallbackResNetModelLabel() {
        // Preferido: etiqueta exacta tal como se carga en populateDlModels()
        final String preferredName = "resnet50_ASL_DOBLE.h5";
        final String preferredLabel = "DL: " + preferredName;
        if (dlModelMap.containsKey(preferredLabel)) return preferredLabel;
        // Búsqueda flexible por nombre de archivo ignorando mayúsculas
        try {
            for (Map.Entry<String, String> e : dlModelMap.entrySet()) {
                String label = e.getKey();
                String path = e.getValue();
                String name = new java.io.File(path).getName();
                if (preferredName.equalsIgnoreCase(name)) return label;
            }
            // Último recurso: cualquier resnet50 asl doble
            for (Map.Entry<String, String> e : dlModelMap.entrySet()) {
                String path = e.getValue();
                String name = new java.io.File(path).getName().toUpperCase();
                if (name.contains("RESNET50") && name.contains("ASL") && name.contains("DOBLE")) {
                    return e.getKey();
                }
            }
        } catch (Exception ignore) {}
        return null;
    }

    private void ensureModelInComboAndSelect(String label) {
        if (label == null) return;
        try {
            DefaultComboBoxModel<String> model = (DefaultComboBoxModel<String>) this.listaModelos.getModel();
            boolean exists = false;
            for (int i = 0; i < model.getSize(); i++) {
                if (label.equals(model.getElementAt(i))) { exists = true; break; }
            }
            if (!exists) {
                // Añadir encabezado de DL si no existe ya
                boolean hasHeader = false;
                String header = "──────── DL MODELS ────────";
                for (int i = 0; i < model.getSize(); i++) {
                    if (header.equals(model.getElementAt(i))) { hasHeader = true; break; }
                }
                if (!hasHeader) model.addElement(header);
                model.addElement(label);
            }
            this.listaModelos.setSelectedItem(label);
        } catch (Exception ignore) {}
    }
}
