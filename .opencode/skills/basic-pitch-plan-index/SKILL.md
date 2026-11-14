---
description: Localiza rápidamente las secciones y fases del plan de
  implementación de basic-pitch-rs. Úsala antes de buscar o modificar
  documentación del plan.
name: basic-pitch-plan-index
---

# Skill: índice del plan de implementación

## Propósito

Ayudar al agente a encontrar rápidamente el archivo correcto del plan de
implementación de `basic-pitch-rs`, sin recorrer todos los documentos ni
confundir el número del archivo con el número real de la fase.

## Regla principal

**Lee este índice primero.** Cada archivo `fase-NN` contiene una fase
del plan, pero su número de archivo no siempre coincide con el número de
fase.

-   El plan comienza en `fase-04-implementacion.md`, que contiene la
    **Fase 0**.
-   Cada archivo posterior representa exactamente una fase.
-   El plan termina en `fase-19-documentacion.md`, que contiene la
    **Fase 15**.
-   El apartado de criterios generales de finalización fue eliminado
    intencionalmente; no busques un archivo para ese apartado.

## Índice de fases

  --------------------------------------------------------------------------------------------------------------
                 Fase del plan Archivo                                                     Contenido / cuándo
                                                                                           consultarlo
  ---------------------------- ----------------------------------------------------------- ---------------------
```
📁 Proyecto Basic-Pitch (Rust Port)
│
├── 📂 docs/
│   ├── 📂 teoria/
│   │   └── 📂 primer paso/
│   │       ├── 📄 fase-04-implementacion.md         [0] Auditoría del modelo oficial: arquitectura activa, capas, constantes, tensores y puntos de comparación.
│   │       ├── 📄 fase-05-conversion-pesos.md       [1] Extracción de pesos, SavedModel, ONNX, conversión a Safetensors y manifiestos.
│   │       └── 📄 fase-06-entorno-rust.md           [2] Workspace Cargo, crates, dependencias y configuración del entorno Rust.
│   │
│   └── 📂 practica/
│       ├── 📄 fase-07-procesamiento-audio.md        [3] Decodificación, mono, remuestreo, ventanas, padding y solapamiento.
│       ├── 📄 fase-08-frontend-cqt.md               [4] CQT, kernels, NormalizedLog y Batch Normalization inicial.
│       ├── 📄 fase-09-harmonic-stacking.md          [5] Desplazamientos armónicos, padding, canales y dimensiones de entrada.
│       ├── 📄 fase-10-arquitectura-neuronal.md      [6] Ramas contour, notes y onset; convoluciones, concatenación, layouts y Candle.
│       ├── 📄 fase-11-validacion-equivalencia.md    [7] Comparación TensorFlow/Candle, tensores intermedios, métricas y tolerancias.
│       ├── 📄 fase-12-inferencia-duracion-arbitraria.md [8] Inferencia por ventanas, reconstrucción temporal y unión de predicciones.
│       ├── 📄 fase-13-posprocesamiento-musical.md   [9] Detección polifónica, umbrales, eventos, duración y Melodia Trick.
│       ├── 📄 fase-14-pitch-bend.md                 [10] Estimación de afinación y conversión de contornos a pitch bend.
│       └── 📄 fase-15-exportacion-midi.md           [11] Eventos MIDI, tracks, canales, Note On/Off, tempo y Pitch Bend.
│
└── 📂 docs/
    └── (raíz de práctica avanzada / cierre)
        ├── 📄 fase-16-integracion-motor.md          [12] API pública, estructuras de resultado, carga del modelo y errores.
        ├── 📄 fase-17-pruebas-automatizadas.md        [13] Pruebas unitarias, integración, regresión y CI.
        ├── 📄 fase-18-rendimiento.md                  [14] Benchmarks, memoria, optimización y backends.
        └── 📄 fase-19-documentacion.md              [15] Documentación técnica, procedencia de pesos, licencias y preparación para integración futura.
        |__ 📄 bugs-resueltos.md                     Resolucion de bugs conocidos en el software documentacion de oro.
```  
  --------------------------------------------------------------------------------------------------------------

## Documentos generales

  ----------------------------------------------------------------------------------------
  Archivo                                              Uso
  ---------------------------------------------------- -----------------------------------
  `docs/teoria/fase-01-objetivo.md`                    Consultar el propósito, alcance y
                                                       límites del proyecto.

  `docs/teoria/fase-02-arquitectura.md`                Consultar el flujo general y la
                                                       separación de módulos.

  `docs/teoria/fase-03-espe-tecnicas.md`   Consultar constantes, dimensiones,
                                                       parámetros y convenciones técnicas.

  `docs/implementacion.md`                             Plan maestro completo. Úsalo para
                                                       contexto global o cuando una
                                                       decisión afecte varias fases.
  ----------------------------------------------------------------------------------------

> Si la ubicación real de `fase-19-documentacion.md` difiere de `docs/`,
> localízalo por nombre dentro de `docs/`; no inventes una ruta
> alternativa.

## Cómo encontrar una sección

1.  Identifica el concepto o tarea que necesitas resolver.
2.  Busca sus palabras clave en la tabla de fases.
3.  Abre únicamente el archivo indicado.
4.  Si necesitas contexto o dependencias, consulta también
    `docs/implementacion.md` y las fases relacionadas.
5.  Antes de modificar una fase, revisa sus referencias a otras fases
    para mantener consistencia.

### Atajos por tema

-   **TensorFlow, ONNX, SavedModel, Safetensors, pesos:** Fase 1.
-   **Python, uv, dependencias y Cargo:** Fase 2.
-   **MP3, WAV, mono, soxr, resampling, ventanas:** Fase 3.
-   **309 bins, CQT, firwin2, kernels, NormalizedLog:** Fase 4.
-   **264 bins, armónicos, shifts:** Fase 5.
-   **Candle, Conv2D, padding, BatchNorm, contour/notes/onset:** Fase 6.
-   **MAE, RMSE, parity, cross-check, referencia:** Fase 7.
-   **172 frames, 142 frames, overlap, unwrap, alineación:** Fase 8.
-   **Umbrales, onset detection, notas polifónicas:** Fase 9.
-   **Vibrato, afinación, pitch bend:** Fase 10.
-   **MIDI, Note On, Note Off, canales, tracks:** Fase 11.
-   **API, biblioteca, transcriber:** Fase 12.
-   **Tests, CI, regresión:** Fase 13.
-   **Benchmarks, CPU, GPU, memoria, optimización:** Fase 14.
-   **README, licencias, documentación y entrega:** Fase 15.

## Instrucciones para el agente

-   No asumas que `fase-10` significa Fase 10 del plan: significa **Fase
    6**.
-   No dupliques contenido de una fase dentro de otra; enlaza o
    referencia el archivo correspondiente.
-   No cambies parámetros técnicos por intuición. Comprueba la fase
    pertinente y la referencia técnica.
-   Si una tarea cruza varias fases, identifica todas las afectadas y
    respeta el orden de dependencias.
-   Si el índice y un archivo parecen contradecirse, inspecciona
    `docs/implementacion.md` y los documentos fuente; señala la
    discrepancia en vez de inventar una resolución.
-   Mantén este índice actualizado si se renombran, mueven, dividen o
    agregan archivos.
