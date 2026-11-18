//! Utilidades compartidas por los tests de integracion.
//!
//! Cada fichero de `tests/` es un crate distinto, y no todos usan todas las
//! funciones de este modulo: de ahi el `allow(dead_code)`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Raiz del repositorio.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Audio de `vocadito_10`.
///
/// Primero se busca el fixture versionado en este repositorio y, si no esta,
/// el clon local del upstream (`basic-pitch/`, que `.gitignore` excluye y por
/// tanto **no existe en un checkout limpio**, como el del CI).
pub fn vocadito_path() -> PathBuf {
    let fixture = repo_root().join("reference/audio_reference/vocadito_10.wav");
    if fixture.exists() {
        fixture
    } else {
        repo_root().join("basic-pitch/tests/resources/vocadito_10.wav")
    }
}

/// Registra que un test se omite por falta de fixtures.
///
/// Si `BASIC_PITCH_REQUIRE_FIXTURES` esta definida (el CI la define), omitir
/// es un **fallo**: un CI que se salta en silencio la validacion contra el
/// upstream da una falsa sensacion de seguridad.
pub fn omitir(motivo: &str) {
    if std::env::var_os("BASIC_PITCH_REQUIRE_FIXTURES").is_some() {
        panic!("faltan fixtures y BASIC_PITCH_REQUIRE_FIXTURES esta definida: {motivo}");
    }
    eprintln!("OMITIDO: {motivo}");
}
