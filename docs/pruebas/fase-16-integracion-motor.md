# Fase 12: integración del motor de transcripción

**Objetivo:** reunir todos los módulos en una API sencilla para utilizar el motor desde otros programas.

La arquitectura deberá evitar que las capas de la red neuronal conozcan detalles del formato de audio o del archivo MIDI.

## 12.1. API pública

Se recomienda exponer una interfaz similar a:

```rust
pub struct Transcriber {
    // Modelo cargado
    // Configuración de inferencia
    // Dispositivo de ejecución
}

impl Transcriber {
    pub fn load(
        model_path: &str,
        device: Device,
    ) -> Result<Self>;

    pub fn transcribe(
        &self,
        audio_path: &str,
    ) -> Result<TranscriptionResult>;
}
```

La API definitiva podrá incorporar opciones adicionales, pero deberá conservar la simplicidad de uso.

## 12.2. Resultado

Se recomienda una estructura similar a:

```rust
pub struct TranscriptionResult {
    pub notes: Vec<NoteEvent>,
    pub midi_data: Vec<u8>,
    pub duration_seconds: f64,
}
```

También podrá existir un resultado de depuración que conserve las salidas originales del modelo.

## 12.3. Carga única del modelo

El modelo deberá cargarse una sola vez y reutilizarse para procesar múltiples archivos.

No se deberán volver a leer los pesos desde Safetensors cada vez que se solicite una transcripción.

Esto reducirá los tiempos de espera y evitará trabajo repetido.

## 12.4. Errores

Se deberán definir errores específicos para:

- Archivo inexistente.
- Formato no compatible.
- Error de decodificación.
- Error de remuestreo.
- Error de dimensiones.
- Peso faltante.
- Configuración incorrecta.
- Error del backend de Candle.
- Fallo de generación MIDI.

Los errores deberán tener mensajes útiles para diagnosticar el problema.

**Entregable de la fase:** biblioteca Rust que permita transcribir audio y obtener los eventos o el archivo MIDI sin conocer los detalles internos.
