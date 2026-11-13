
# 1. Objetivo general

Desarrollar una implementación en Rust capaz de recibir archivos de audio, procesarlos mediante el mismo flujo de preprocesamiento que utiliza Basic Pitch, ejecutar su arquitectura neuronal con Candle y generar una transcripción musical exportable a MIDI.

El proyecto debe reproducir tanto la red neuronal como los componentes externos que hacen posible su funcionamiento. No basta con trasladar las capas convolucionales: también es necesario reproducir correctamente la transformación Constant-Q (CQT), la normalización logarítmica, la acumulación de armónicos, el procesamiento de ventanas de audio, la reconstrucción temporal de las predicciones y el algoritmo de detección de notas.

La implementación deberá conservar el comportamiento del modelo original en la medida que lo permitan las diferencias numéricas entre frameworks.

## 1.1. Alcance

El proyecto contempla:

- Implementar en Rust la arquitectura neuronal activa de Basic Pitch.
- Utilizar los pesos preentrenados originales.
- Convertir los pesos al formato Safetensors sin alterar sus valores.
- Cargar y ejecutar los pesos mediante Candle.
- Implementar el preprocesamiento de audio necesario.
- Reproducir el procesamiento CQT y Harmonic Stacking.
- Procesar audios de distintas duraciones mediante ventanas superpuestas.
- Obtener las salidas de notas, comienzos de notas y contornos de frecuencia.
- Reproducir el posprocesamiento que transforma las predicciones en eventos musicales.
- Calcular y conservar información de pitch bend.
- Generar archivos MIDI.
- Verificar los resultados comparándolos con la implementación oficial de referencia.

## 1.2. Fuera del alcance inicial

Para mantener el proyecto controlado, esta primera implementación no incluirá:

- Entrenamiento de la red neuronal.
- Fine-tuning del modelo.
- Modificación de los pesos originales.
- Una interfaz gráfica.
- Un editor de piano roll.
- Un editor de partituras.
- Síntesis de instrumentos virtuales.
- Generación musical mediante inteligencia artificial.
- Exportación de WAV, MP3 o MP4 a partir del MIDI.

Estas funcionalidades podrán formar parte del producto completo posteriormente. La prioridad inicial será obtener un motor de transcripción funcional, verificable y reutilizable.

## 1.3. Principios técnicos

1. **Fidelidad antes que optimización:** primero se reproducirá el comportamiento original; después se mejorará el rendimiento.
2. **Pesos originales:** los valores numéricos de los pesos no deberán modificarse.
3. **Sin Python en producción:** Python solo servirá como herramienta de desarrollo y validación.
4. **Separación de responsabilidades:** audio, CQT, red neuronal, posprocesamiento y MIDI tendrán módulos independientes.
5. **Pruebas por etapas:** cada componente deberá validarse antes de integrarse con el siguiente.
6. **Resultados reproducibles:** se registrarán las versiones, configuraciones, dimensiones y procedencia de los recursos utilizados.

