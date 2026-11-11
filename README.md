# Agente TAG (Table-Augmented Generation) sobre Exoplanetas con DuckDB y Groq

**Instituto Tecnológico de Ocotlán**  
**Materia:** Inteligencia Artificial (SCC-1012)  
**Profesor:** Juan Martín González Razo  
**Alumno:** Osbaldo Ismael Vázquez Hernández  
**Número de Control:** 23630176  

---

## 1. Descripción General y Arquitectura TAG

Este repositorio contiene la implementación de un agente de **Generación Aumentada por Tablas (TAG, *Table-Augmented Generation*)** sobre el catálogo astrofísico de parámetros planetarios compuestos (`pscomppars`) del **NASA Exoplanet Archive**. El sistema emplea **DuckDB** como motor analítico columnar en modo de solo lectura y **Groq (`openai/gpt-oss-120b`)** como modelo de lenguaje de baja latencia, **sin depender de frameworks de abstracción intermedios como LangChain**.

A diferencia de los sistemas basados en recuperación vectorial (*RAG*), donde el texto se fragmenta y codifica en vectores de similitud, la arquitectura **TAG** delega el razonamiento cuantitativo a un motor relacional analítico mediante **3 etapas explícitas y auditables**:

1. **Etapa 1 — Síntesis de SQL (*Text-to-SQL*):** El modelo de lenguaje inspecciona únicamente el esquema tipado de `pscomppars` y su diccionario de unidades físicas (sin acceder a las filas de la tabla) para sintetizar una consulta `SELECT` en dialecto DuckDB con temperatura determinista (`0.0`). Incorpora un **bucle de autorreparación guiado por errores** de hasta 3 intentos en caso de fallo sintáctico o de enlace de columnas (*Binder Error*).
2. **Etapa 2 — Ejecución Analítica Segura en DuckDB:** La sentencia SQL es verificada previamente por un validador determinístico de seguridad y ejecutada sobre una conexión `duckdb.connect(..., read_only=True)`, registrando latencia en milisegundos, metadatos de columnas y registros recuperados.
3. **Etapa 3 — Redacción Fundamentada (*Grounded Generation*):** El modelo redacta la respuesta técnica en español basándose **exclusivamente** en la evidencia tabular devuelta por DuckDB. Incluye un cortocircuito determinístico contra alucinaciones cuando la consulta devuelve `0 filas` o cuando el usuario solicita atributos inexistentes en el esquema.

---

## 2. Estándares de Ingeniería y Reproducibilidad

El desarrollo de [`tag_exoplanetas_duckdb.ipynb`](./tag_exoplanetas_duckdb.ipynb) adopta las normas de la **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)** junto con las prácticas de **Ingeniería de Investigación (*Research Engineering*)** utilizadas en los repositorios abiertos de Google Research y Google DeepMind:

- **Tipado estático explícito (`PEP 484` / `PEP 604`):** Anotaciones estrictas en todas las firmas de funciones, estructuras y colecciones (`from __future__ import annotations`).
- **Documentación estructurada (*Google-style Docstrings*):** Especificación formal de `Args`, `Returns` y `Attributes` en cada función y clase del flujo de datos.
- **Configuración hermética e inmutable (`@dataclass(frozen=True)`):** Centralización de rutas, hiperparámetros, límites de contexto y semilla aleatoria (`SEED = 42`) en el objeto inmutable `ConfiguracionTAG`.
- **Contratos de datos validados (`pydantic.BaseModel` con `frozen=True`):** Modelado explícito de las entradas y salidas de cada etapa (`SalidaSintesisSQL`, `IntentoSQL`, `ResultadoEtapa2`, `ResultadoTAG`).
- **Verificación en cuaderno (*In-Notebook Testing & Red-Teaming*):** Pruebas unitarias deterministas con `assert` para los guardrails de seguridad antes de invocar al modelo, seguidas de un benchmark reproducible de 10 niveles y pruebas de estrés (*red-teaming*).

---

## 3. Esquema de Datos (`pscomppars`) y Plan de Contingencia

La tabla `pscomppars` (*Planetary Systems Composite Parameters*) almacena una fila por exoplaneta confirmado con las siguientes **8 columnas**:

| Columna | Tipo DuckDB | Descripción Física y Unidades |
| :--- | :--- | :--- |
| `pl_name` | `VARCHAR` | Nombre estándar del exoplaneta (ej. `'51 Peg b'`, `'TRAPPIST-1 e'`). |
| `hostname` | `VARCHAR` | Nombre de la estrella anfitriona (ej. `'51 Peg'`, `'TRAPPIST-1'`). |
| `disc_year` | `BIGINT` | Año de publicación del descubrimiento (ej. `1995`, `2017`). |
| `discoverymethod` | `VARCHAR` | Método de descubrimiento en inglés (`'Transit'`, `'Radial Velocity'`, `'Imaging'`, `'Microlensing'`, etc.). |
| `pl_rade` | `DOUBLE` | Radio planetario en radios terrestres ($R_\oplus$). Admite `NULL`. |
| `pl_masse` | `DOUBLE` | Masa planetaria en masas terrestres ($M_\oplus$). Admite `NULL`. |
| `pl_orbper` | `DOUBLE` | Periodo orbital en días terrestres. Admite `NULL`. |
| `sy_dist` | `DOUBLE` | Distancia desde el Sol al sistema planetario en pársecs ($\text{pc}$). Admite `NULL`. |

### Estrategia de Adquisición Resiliente (Plan A + Plan B)
- **Plan A (API TAP oficial de NASA/Caltech):** Descarga directa en formato CSV mediante el protocolo *Table Access Protocol* (`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`).
- **Plan B (Contingencia determinista ante mantenimiento o modo sin conexión):** Si el servidor de IPAC/Caltech devuelve error de disponibilidad (`HTTP 503`) o no existe conexión de red, el sistema sintetiza de manera determinista (`SEED = 42`) un catálogo local de 550 registros que integra exoplanetas reales verificados (*Proxima Centauri*, *TRAPPIST-1*, *51 Pegasi*, *Kepler-90*, *HD 10180*, *Kepler-186*, *TOI-700*, *PSR B1257+12*, entre otros) con una muestra astrofísica consistente.

---

## 4. Justificación Técnica del Modelo LLM en Groq

Se seleccionó el modelo **`openai/gpt-oss-120b`** ejecutándose sobre la infraestructura de inferencia de Groq por tres criterios técnicos:
1. **Precisión en síntesis de SQL analítico (*Text-to-SQL*):** La generación de expresiones comunes de tabla (`WITH`), filtros de grupo (`HAVING`), subconsultas escalares comparativas y cláusulas de ordenamiento con control de nulos (`IS NOT NULL`, `NULLS LAST`) requiere un modelo de alta capacidad (120B parámetros) para alcanzar un índice de éxito del 100% en el primer intento (*pass@1*).
2. **Soporte nativo de salidas estructuradas (`json_object`):** Garantiza el cumplimiento del esquema `SalidaSintesisSQL` validado por `Pydantic`.
3. **Baja latencia de inferencia:** Permite ejecutar el flujo completo de 3 etapas (incluyendo eventuales ciclos de autorreparación) con tiempos de respuesta reducidos.

---

## 5. Arquitectura de Seguridad SQL (Defensa en Profundidad)

1. **Validador Léxico y Sintáctico (`validar_consulta_sql`):**
   - Exige que toda consulta inicie exclusivamente con `SELECT` o `WITH`.
   - Bloquea comentarios SQL (`--`, `/* */`) y múltiples sentencias separadas por `;` (*stacked queries*).
   - Rechaza verbos de mutación, definición o administración (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `REPLACE`, `COPY`, `ATTACH`, `DETACH`, `PRAGMA`, `INSTALL`, `LOAD`, `CALL`, `EXPORT`, `IMPORT`).
   - Bloquea funciones de lectura de archivos del sistema propias de DuckDB (`read_csv`, `read_csv_auto`, `read_parquet`, `read_json`, `glob`, `sniff_csv`) para impedir la lectura de archivos locales fuera de la tabla autorizada `pscomppars`.
2. **Aislamiento en el Motor (`read_only=True`):**
   - Toda consulta del agente se ejecuta sobre una conexión `duckdb.connect(..., read_only=True)`, impidiendo modificaciones a nivel de sistema de archivos.
3. **Bucle de Autorreparación Guiado por Errores (`max_reintentos_sql = 3`):**
   - Si una consulta es rechazada por el validador o genera una excepción `duckdb.Error`, el agente retroalimenta el mensaje de error exacto al modelo para corregir la consulta automáticamente.
4. **Salvaguarda Determinista Anti-Alucinación:**
   - Si la ejecución en DuckDB devuelve `0 filas` (o valores nulos en todas las columnas agregadas), el sistema omite la llamada generativa final y emite una notificación explícita de ausencia de registros.

---

## 6. Estructura del Repositorio

* [`tag_exoplanetas_duckdb.ipynb`](./tag_exoplanetas_duckdb.ipynb): Cuaderno Jupyter principal con la implementación completa, pruebas unitarias de seguridad, benchmark de 10 preguntas de dificultad creciente, tabla de telemetría y pruebas de robustez (*red-teaming*).
* [`requirements.txt`](./requirements.txt): Especificación de dependencias del proyecto (`duckdb`, `groq`, `pydantic`, `python-dotenv`, `jupyter`).
* [`.env.example`](./.env.example): Plantilla de configuración para variables de entorno (`GROQ_API_KEY`, `MODELO_GROQ`).
* [`.gitignore`](./.gitignore): Reglas de exclusión para proteger credenciales locales (`.env`), el entorno virtual (`.venv/`) y los artefactos de datos generados (`datos/`).

---

## 7. Instalación y Reproducción

```bash
# 1. Clonar el repositorio e ingresar al directorio
git clone https://github.com/osbas/tag-exoplanetas.git
cd tag-exoplanetas

# 2. Crear y activar el entorno virtual de Python
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar las dependencias
.venv/bin/pip install -r requirements.txt

# 4. Configurar las variables de entorno
cp .env.example .env
# Editar el archivo .env e incorporar una clave válida en GROQ_API_KEY

# 5. Iniciar Jupyter Notebook
.venv/bin/jupyter notebook tag_exoplanetas_duckdb.ipynb
```
