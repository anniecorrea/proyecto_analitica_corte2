# Uso de IA

## Integración y evaluación PENGWIN

- **Prompt:** Auditar `evaluacion.py` contra las interfaces reales de los notebooks y `deteccion_grid.py`; verificar clases, checkpoint y dependencias; corregir únicamente el evaluador y probarlo, sin inventar artefactos ni umbrales.
- **Sugerencias de IA:** Permitir la carga segura del objeto `TorchVersion`, exigir que los umbrales estén guardados en el checkpoint, validar el mapeo de clases y calcular una máscara de scores por detección para filtrar las cajas de segmentación.
- **Cambios aplicados:** Se verificaron las firmas y convenciones en el código fuente, se corrigió el filtro de cajas que fallaba al indexar tensores y se añadieron validaciones de rutas, clases y metadatos obligatorios del checkpoint. El NMS usado en pruebas fue `deteccion_grid.nms_por_clase`.
- **Análisis crítico:** Las pruebas sintéticas confirmaron el recorrido de evaluación y la escritura de métricas, tabla, ejemplos y latencias en CPU/CUDA; esos valores no describen el modelo real. El checkpoint C disponible carga y ejecuta el pipeline, pero carece de `inference_defaults`, por lo que no se deben inferir resultados reales hasta regenerar/exportar el checkpoint con esos metadatos. No se encontró checkpoint ni paquete SAM; esa comparación queda pendiente.

## Cortes sin pelvis: ajuste fino con negativos (12 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Daniela en la app de Claude con el repositorio y los resultados de `val` como contexto.
- **Prompt (textual, el principal):** "tengo un problema sobre unos cortes, el modelo se entrenó casi solo con cortes que sí tienen pelvis. Por eso, cuando analiza cortes sin pelvis, a veces cree detectar una y genera falsos positivos.". Después: "qué estaremos implementando para tener todo listo? recuerda que este proyecto requiere mas calidad que cantidad y es necesario sacrificar tiempo con tal de tener algo bueno".
- **Qué salió de la IA:** el diagnóstico con los datos de `scores_cortes_val.csv` (310 de 394 cortes sin pelvis con alguna caja; 190 pasan el umbral 0,45; los falsos positivos aparecen también lejos de la pelvis), `src/cortes_vacios.py`, `tests/test_cortes_vacios.py`, `notebooks/Negativos_ajuste.ipynb`, la celda de comparación 3D para `salida_3d.ipynb` y la regla de selección escrita antes de entrenar.
- **Corrección encontrada en el camino:** `src/definiciones_notebooks.cargar_definiciones` perdía los decoradores (`ast.get_source_segment` empieza en la línea del `def`), así que una función como `evaluate`, marcada con `@torch.no_grad()`, se cargaba calculando gradientes. Se corrigió con `_segmento` y se agregó `tests/test_definiciones_notebooks.py`. Ningún notebook existente cargaba funciones decoradas, así que no cambia sus resultados.
- **Qué verificó la IA y qué no:** las funciones de `src/cortes_vacios.py` se probaron sin GPU (23 pruebas, más 3 del cargador de definiciones), incluida una que reproduce exactamente `umbral_corte.json` con los scores reales de `val`. El notebook **no** se ejecutó en el entorno de la IA (no había torch ni datos completos): se revisó que todos los nombres que usa existan en los notebooks del equipo y que la sintaxis sea válida. La primera ejecución con `SMOKE_TEST = True` es la prueba real.
- **Revisión tras el cambio a entrada 2.5D (commits de Juan Hoyos del 12 de octubre):** el modelo nuevo usa los cortes z-1, z, z+1 y centros "interior". Con sus scores de `val`, 229 de 394 cortes sin pelvis tienen alguna caja y 212 pasan el umbral recalibrado de 0,08: el problema sigue. El notebook se adaptó para leer `IN_CHANNELS` y `CENTER_MODE` de `config.json` (sirve con el modelo de 1 o de 3 canales) y las pruebas ya no fijan los números del modelo anterior, sino que los comparan con `umbral_corte.json`.
- **Decisiones que la IA descartó tras revisar los datos:** el filtro de "tramo principal en z" no se adopta por defecto. Con los scores de `val` su efecto no es estable (con un hueco tolerado de 2 cortes quita 93 de 190 falsos positivos; con 5 solo quita 53) y en el caso 035 los falsos positivos están pegados a la pelvis, así que no los quitaría. Queda como evaluación informativa en la sección 9 del notebook. Con el modelo 2.5D el filtro rinde más (con hueco 2 pasa de 212 a 93 cortes vacíos, perdiendo 3 cortes con pelvis; con hueco 10, a 130 sin perder ninguno), pero sigue dependiendo mucho del hueco elegido, así que se mantiene como opción a decidir con el resultado del ajuste.
- **Cambios hechos a mano por el equipo:** Juan Hoyos ejecutó el notebook en su RTX 5060 (prueba corta y corrida real) y la comparación 3D. La primera versión de la celda de comparación 3D que propuso la IA agotó la RAM (guardaba los 5 volúmenes con sus salidas a la vez) y se corrigió para procesar un caso a la vez. El equipo decidió no adoptar este modelo (ver la entrada siguiente).
- **Análisis crítico:** la regla se cumplió (ganó "todos", época 2): los cortes vacíos que pasan bajaron de 212 a 172 de 394 sin cambio en mAP (0,864) ni Dice (0,944 frente a 0,943). Pero en 3D, con la configuración del modelo original, el F1 por fragmento bajó de 0,667 a 0,638 (20 fragmentos frente a 18). La IA afirmó al principio que el filtro de la salida 3D ya eliminaba casi todos los cortes vacíos; al revisarlo con los datos se vio que no (163 de 394 seguían pasando) y se corrigió. Limitación: la época y el umbral se eligieron con los mismos 5 casos de `val`.

## Cortes sin pelvis: entrenamiento desde cero, comparación final y control de pelvis (13 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Daniela en la app de Claude con el repositorio y los resultados del ajuste fino como contexto.
- **Contexto y prompt:** el profesor indicó que el sistema se probará con un estudio `.mha` que puede tener pelvis o no, así que hay que usar negativos. Prompt: "seria interesante re entrenar desde cero con negativos en todas las etapas". Después se acordó en equipo que el sistema debe cumplir las dos cosas: separar bien los fragmentos con pelvis y no inventar fragmentos sin pelvis.
- **Qué salió de la IA:** `notebooks/Entrenamiento_negativos_completo.ipynb` (mismo entrenamiento A, B y C del original, con los 5.698 negativos de train), `notebooks/Comparacion_final_negativos.ipynb` (original, ajustado y desde cero en 3D, con la salida 3D reajustada para cada uno y los tramos sin pelvis de val como estudios separados), `src/control_pelvis.py` y su uso en `app/inferencia.py` y `app/paginas/prediccion.py`, la función `regla_modelo_final` y `tramos_sin_pelvis` en `src/cortes_vacios.py`, y `tests/test_control_pelvis.py`.
- **Regla acordada antes de entrenar:** un candidato es aceptable si (1) no baja más de 0,01 en mAP ni en Dice frente al original y conserva el 99 % de los cortes con pelvis, (2) su F1 3D no queda más de un fragmento (0,04) por debajo del original y (3) el sistema con el control entrega 0 fragmentos en los tramos sin pelvis. Gana el de mayor F1 3D; con empate de un fragmento, el que invente menos fragmentos sin el control. La IA propuso una primera versión que el equipo corrigió: incluía al original como candidato (contradecía la decisión de usar negativos), su margen de empate (0,02) era menor que un fragmento y ponía el caso sin pelvis por encima del caso con pelvis.
- **Cambio de la regla (13 de octubre, 10 p. m.):** el entrenamiento desde cero bajó la mAP@0.50 en 0,0106 frente al original (0,854 frente a 0,864) y redujo los cortes vacíos que pasan de 212 a 51. Con la tolerancia original (0,01) quedaba descartado por 6 milésimas. Daniela propuso que 0,01 es una pérdida insignificante; la IA confirmó que la consigna no exige una regla de selección (solo justificar las decisiones y cumplir las metas de la sección 5, como mAP@0.50 ≥ 0,65) y propuso ampliar la tolerancia a 0,02, con base en la variación normal entre épocas (0,847-0,866 en la etapa C), en lugar de un valor hecho a la medida. El equipo lo aprobó **antes** de correr la comparación final en 3D. Las demás condiciones no cambiaron.
- **Control de "pelvis presente":** umbral de 80 mm de bloque continuo con score ≥ 0,2, fijado con los scores guardados de val antes de comparar modelos: con pelvis el bloque mide 158 mm o más (modelo ajustado) y 208 mm o más (original); en los tramos sin pelvis, 32 mm como mucho. Una prueba automática verifica esto con los datos reales.
- **Qué verificó la IA y qué no:** las funciones nuevas se probaron sin GPU (40 pruebas en total). Los notebooks **no** se ejecutaron en el entorno de la IA (sin torch ni volúmenes): se revisó que todos los nombres que usan existan en los notebooks del equipo. La primera ejecución con `SMOKE_TEST = True` en la GPU de Juan Hoyos es la prueba real.
- **Cambios hechos a mano por el equipo:** Juan Hoyos ejecutó el entrenamiento desde cero (127 min) y la comparación final. El equipo corrigió la primera regla propuesta por la IA, amplió la tolerancia de mAP a 0,02 antes de la comparación 3D y, después de ver los resultados, **decidió apartarse de la regla y adoptar el entrenamiento con negativos desde cero** (ver análisis). Con esa decisión, el equipo entrenó los tres modelos finales (axial, coronal y sagital) con negativos desde la etapa A, uno por cada tres positivos y el 60 % cerca de la pelvis, y con una selección de checkpoint que penaliza los cortes vacíos que pasan (`notebooks/Entrenamiento_multitarea_planos.ipynb`). El axial final deja pasar 0 de 394 cortes vacíos de val, con mAP@0.50 de 0,912 frente a 0,864 del original.
- **Análisis crítico:** resultados en val (salida 3D reajustada para cada modelo):

  | Modelo | mAP@0.50 | Dice | Cortes vacíos que pasan (de 394) | Fragmentos (reales 27) | Correctos | F1 3D |
  |---|---|---|---|---|---|---|
  | Original | 0,864 | 0,944 | 212 | 18 | 15 | 0,667 |
  | Ajustado | 0,864 | 0,943 | 172 | 19 | 15 | 0,652 |
  | Desde cero | 0,854 | 0,941 | 51 | 21 | 15 | 0,625 |

  La regla elegía el ajustado: el desde cero quedó 0,0017 por fuera de la tolerancia de F1 3D (0,042 frente a 0,04). **El equipo decidió apartarse de la regla y adoptar el modelo entrenado desde cero** porque (1) el sistema se evaluará con estudios que pueden no tener pelvis y este modelo reduce los cortes vacíos que pasan de 212 a 51 (−76 %), frente a 172 del ajustado; (2) con pelvis encuentra los mismos 15 fragmentos que los otros dos y la diferencia son 2 fragmentos falsos en 2 de 5 casos (035 y 067), dentro del ruido de 5 casos; (3) cumple todas las metas de la consigna. La IA presentó las dos opciones y advirtió que, si se elegía el desde cero, había que reportarlo como una decisión que se aparta de la regla y no como resultado de ella. También queda registrado que la tolerancia de mAP se amplió de 0,01 a 0,02 cuando un candidato quedaba fuera por poco: se aprobó antes de la comparación 3D, pero es un cambio de regla a mitad del proceso.

  Sin pelvis, los tres modelos entregaron 0 fragmentos en los 10 tramos sin pelvis de val, incluso sin el control, porque son tramos cortos (17 a 80 mm) que la salida 3D ya descarta: esa prueba no distingue entre modelos. El control de "pelvis presente" rechazó los 10 tramos y aceptó los 5 casos con pelvis. Limitaciones: una sola semilla; los mismos 5 casos de val se usaron para reajustar la salida 3D y para elegir; los tramos sin pelvis son anatomía vecina de la pelvis (fémur, columna) y no hay estudios largos sin pelvis en el dataset, que es el caso en que la ventaja del modelo desde cero debería notarse más.

## Verificación de los modelos por plano con el control de pelvis y la salida 3D (16 y 17 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Daniela en la app de Claude con el repositorio como contexto.
- **Prompt:** "Quiero que vuelvas a revisar el repositorio y me digas si hoyos ya actualizó". Juan Hoyos pidió no modificar nada del repositorio, así que la IA entregó el script para que él lo ejecutara.
- **Qué salió de la IA:** `src/verificar_planos_val.py`. (1) Prueba el control de pelvis con los tres modelos finales sobre los volúmenes de val y sobre recortes sin pelvis. (2) Repite la salida 3D con cuatro variantes (actual, sin absorción de trozos pequeños, umbral de centros más bajo y ambas) para ver si la fusión de conminutos viene del armado 3D o del modelo.
- **Cambios hechos a mano por el equipo:** Juan Hoyos lo ejecutó y lo agregó a su corrida de pendientes. Los resultados están en `resultados/metricas/verificacion_planos/`.
- **Resultados:** el control aceptó los 15 volúmenes con pelvis (bloque mínimo de 105 mm) y rechazó los 30 recortes sin pelvis (bloque máximo de 45 mm) sin cambiar sus parámetros. Sin el control, los modelos coronal y sagital habrían entregado 10 fragmentos inexistentes. Con las variantes del armado 3D aparecen como mucho 3 de 12 conminutos, a costa de pasar de 16–23 a 26–120 fragmentos predichos frente a 27 reales: la fusión viene del decoder, no del armado 3D.
- **Análisis crítico:** error de la IA: en un primer análisis sobrestimó el problema de las distancias. Al revisar los datos se vio que 11 de los 12 conminutos de val están en contacto con su fragmento principal (distancia real de 0 mm), así que fusionarlos da la distancia correcta. Limitación: son solo 5 casos de val y los recortes sin pelvis son de anatomía vecina, no estudios de otras regiones del cuerpo.

## Informe técnico (17 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Daniela en la app de Claude con el repositorio, la consigna y cinco artículos de referencia como contexto.
- **Prompts (textuales, los principales):** "Listo, ahora preparémonos para hacer el informe que tiene que tener máximo 4 páginas"; "cree una nueva carpeta que dice pdfs importantes", "me gustaria que hicieras un grafico como estos acorde a lo que se hizo"; "consideras que hasta el momento lo que tenemos del informe es coherente con lo que tenemos en el repositorio?". Para la revisión, Juan Hoyos usó Claude Code con el repositorio: "Revisa el Proyecto2-Analitica.docx y dime si es completo con lo que pide Proyecto_Corte2_PENGWIN.docx es decir, si este es coherente con lo que se ha echo, explica bien todo".
- **Qué salió de la IA:**
  - borradores de las secciones, con las cifras tomadas de los CSV y JSON de `resultados/metricas/`;
  - la Fig. 1, que combina el diagrama del equipo con un esquema nuevo del sistema;
  - la Fig. 2 (matriz de confusión de pertenencia) y la Fig. 3 (galería de los cinco peores casos);
  - la propuesta de referencias.
- **Cambios hechos a mano por el equipo:** Daniela escribió y editó el informe en Word con el formato de la plantilla, eligió la estructura, recortó texto para el límite de páginas y descartó una versión de la Fig. 1 propuesta por la IA. Juan Hoyos revisó el informe contra el repositorio y encontró afirmaciones que no coincidían con lo hecho; se corrigieron todas.
- **Análisis crítico:** acierto: la mayoría de las cifras coincidían exactamente con el repositorio, porque la IA las leyó de los archivos de resultados y no de memoria. Errores de la IA detectados en la revisión:
  - usó la latencia de una medición anterior (24,8 y 95,3 ms en lugar de 28,9 y 102,5 ms);
  - escribió que el transfer learning usó fotos a color, cuando fueron en escala de grises;
  - citó el umbral de centros de la app (0,55) en lugar del de la evaluación 2D (0,25);
  - dijo que la etapa D era de 10 épocas, cuando son hasta 40 con parada temprana;
  - describió la cabeza de pertenencia como la que usa la app, que en realidad asigna el hueso con la salida semántica;
  - afirmó que todos los conminutos se fusionan, cuando algunos no se encuentran;
  - calificó la ablación con más fuerza de la que tienen los intervalos de confianza.

  Las referencias de la arquitectura (YOLO, CBAM, Panoptic-DeepLab y U-Net) las propuso la IA como el origen conocido de cada idea. **[CONFIRMAR con el equipo que fueron las consultadas y borrar esta nota]** Limitación: la IA no puede verificar el formato final del PDF; eso se revisó a mano.

## Auditoría del proyecto frente a la consigna (16 y 17 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Juan Hoyos en Claude Code (extensión de VS Code) con el repositorio y la consigna como contexto.
- **Prompts (textuales, los principales):** "Tu trabajo va a ser analizar todo lo que esta dentro de proyecto_analiticaa_corte2 y con baase al documento Proyecto_Corte2_PENGWIN.docx vas a revisar que se hace a cada semana y decir si se cumple lo que se pide para cada semana". Después: "que esta compelto, que esta parcial y que esta sin hacer, tambien rankealo por gravedad. es decir, si eso no esta bien hecho o no se ha empezado atrasa algo?".
- **Qué salió de la IA:** el estado de cada entregable por semana y una lista de pendientes ordenada según cuánto bloqueaba al resto: la ablación, el transfer learning, las distancias predicción frente a ground truth en test y la comparación con SAM.
- **Cambios hechos a mano por el equipo:** Juan Hoyos fijó el orden de trabajo de los días siguientes con este ranking y repitió la auditoría al final de cada bloque de trabajo.
- **Análisis crítico:** el ranking sirvió para priorizar, pero depende de cómo la IA interpreta la consigna; los puntos dudosos se revisaron contra el documento original. Ante las métricas que no alcanzan la meta, el equipo decidió explicar por qué no se alcanzan en lugar de forzar resultados ("hay cosas que no llegan a lo esperado, pero no estan lejos de la meta, toca es explicar por que no logro llegar, cierto?").

## Etapa B frente a C y ablación de CBAM y transfer learning (17 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Juan Hoyos en Claude Code con el repositorio y la consigna como contexto.
- **Prompts (textuales, los principales):** "Evaluar entre modelos del mismo plano (B y C) para revisar en test si el fine tuning que se hace en la etapa C si mejora los resultados o no, para hacerlo mas automatico, dime que te parece si se hace un intervalo de confianza, es decir, si C no supera a B en X porcentaje se escoge B para ahorrar tiempo de computo"; "Seria correr el CBAM y el Transfer learning, revisa si hay cortes viejos guardados sino dime que es lo mejor en este caso para hacer un buen trabajo y cumplieno las reglas"; "porque preentrenar algo que es para mascotas con algo de pelvis?"; "Vamos a llegar hasta B por cuestiones de tiempo […] No se va a hacer la instancia. La idea es que todo quede bien organizado y automatizado para dejarlo corriendo sin supervison".
- **Qué salió de la IA:** la regla B/C con bootstrap pareado por paciente (IC 95 %, margen de 0,01: C solo se elige si lo supera con un intervalo que no incluya ese margen); `src/preentrenar_backbone_pets.py` (preentrenamiento de FundidoraPC en Oxford-IIIT Pet en escala de grises, 60 épocas, exactitud 0,644); `src/ablacion.py` con las variantes `sin_cbam`, `con_tl` y `sin_cbam_con_tl` hasta la etapa B; `src/resumen_ablacion.py`, y un script para correr todo sin supervisión.
- **Cambios hechos a mano por el equipo:** Juan Hoyos propuso la regla B/C, cuestionó el uso de fotos de mascotas (la IA explicó que la consigna pide preentrenar el modelo propio y que Pets aporta filtros de bajo nivel, no anatomía), recortó el alcance hasta la etapa B y sin instancias, y ejecutó todas las corridas en su RTX 5060.
- **Análisis crítico:** C no superó a B de forma relevante en ningún plano y en el coronal B fue mejor (F1 de pertenencia 0,853 frente a 0,819 en test). La IA preparó el cambio de todo el proyecto a la etapa B, pero por tiempo el equipo decidió no regenerar las métricas y lo revirtió: la evaluación por plano se reporta con los modelos finales (etapa C) y la fusión 3D de la app usa los modelos B, como en `notebooks/Test_final_fusion_B.ipynb`. Quitar CBAM bajó cerca de 0,008 y el transfer learning no aportó; ambas diferencias son pequeñas frente a los intervalos y no se presentan como concluyentes. Errores de la IA: las primeras pruebas cortas fallaron (`ZeroDivisionError` de OneCycleLR y un `NameError` en el espacio de nombres del notebook) y, al probar `--help`, ejecutó scripts sin argparse que regeneraron figuras, que se restauraron con git. Limitación: 20 pacientes de test y una sola semilla.

## Despliegue local y en AWS, fusión 3D y advertencia de pelvis atípica (17 de octubre de 2026)

- **Modelo de IA:** Claude (Anthropic), modelo configurado `claude-opus-5-5`, usado por Juan Hoyos en Claude Code con el repositorio como contexto.
- **Prompts (textuales, los principales):** "Vamos a hacer el despliegue en Cloudflare, lo probe en AWS y funciona bien, pero dejemos un respaldo que funcione de forma local, asi que adapta para que funcione en ambos entornos, ademas de eso, como se guarda la imagen en AWS, ya que me comentan que dependiendo de como se haga ese proceso puede consumir mas costo o no"; "me gustaria saber si es posible como poner un warning si la imagen que se entrega no es una pelvis humana, supongamos que es una pelvis de un animal parecida a la humana […] creo que seria buena idea darle un valor agregado"; "Añadirle un intervalo de error, ya que pueden haber pelvis que se salgan de los parametros del entrenamiento no crees?"; "Con base a notebooks/Test_final_fusion_B.ipynb haga la reconstruccion 3D usando los 3 planos distintos (sagital, coronal y axial) para el despliegue reemplazando la reconstruccion axial que se esta usando actualmente"; "la consigna pide algo de segmentacion por fragmentos, Dice e IoU por fragmentos?".
- **Qué salió de la IA:**
  - `deploy/instalar.sh` (servicios systemd y Git LFS solo para los modelos usados), `deploy/local.ps1` y `deploy/local.bat`, la variable `PENGWIN_DEVICE` y una explicación de los costos de almacenamiento en AWS;
  - `src/plausibilidad_pelvis.py` con dos niveles: una nota si una medida sale del rango observado y una advertencia si sale de un intervalo de tolerancia en escala logarítmica (99,9 % de cobertura, 99 % de confianza);
  - `predecir_fusion` en `app/inferencia.py`: suma las probabilidades de los tres planos, toma el argmax, separa componentes conexos por hueso (≥ 500 mm³) y decide el control de pelvis por mayoría de 2 de 3 planos;
  - `src/evaluar_fusion_3d.py` y las tarjetas de Dice e IoU por fragmento 3D en el dashboard.
- **Cambios hechos a mano por el equipo:** Juan Hoyos propuso la advertencia y su intervalo de error, probó el despliegue en AWS y en local y pidió reemplazar la reconstrucción axial por la fusión.
- **Análisis crítico:** la fusión se verificó vóxel a vóxel contra el notebook (diferencia 0). En test, el Dice 3D por región es 0,973 y por fragmento 3D el Dice es 0,511 y el IoU 0,479 (recall 0,553, precisión 0,969): casi no inventa fragmentos, pero pierde cerca de la mitad de los reales; los conminutos encontrados pasan de 0 a 3 de 54. La advertencia dio 0 avisos en 100 análisis de pelvis humanas y detectó 125 de 125 casos alterados. Errores de la IA: la primera versión de la advertencia daba falsas alarmas por la asimetría entre coxales (se quitó ese criterio); el primer intervalo (99 % / 95 %) era más angosto que el margen anterior y se amplió; en Windows `localhost` resolvía a IPv6 y se cambió por `127.0.0.1`; y en la fusión hubo un choque de nombre de variable (`lote`) que detectaron las pruebas. Limitación: no hay estudios reales de animales; los casos atípicos son simulados.

## Annie Correa — segmentación de instancias, CBAM, salida 3D y SAM

**Modelo de IA:** Claude (Anthropic), usado por Annie en una misma conversación larga con acceso a los notebooks y archivos que ella subía. En la parte final (11 al 17 de octubre de 2026) el modelo configurado fue `claude-sonnet-5-5`; el modelo que atendió cada turno pudo variar. [CONFIRMAR con Annie el modelo de las partes anteriores.]

**Cómo se redactó esta sección:** los prompts de los avances 3 se transcribieron del registro de la conversación, sin corregir ortografía. Donde el prompt llevaba archivos adjuntos (notebooks, errores, tablas), se indica entre paréntesis. Lo que no consta en el registro está marcado con [COMPLETAR] o [CONFIRMAR].

---

### Avance 1 (semana 8) — [COMPLETAR]

[COMPLETAR si Annie usó IA en su parte del Avance 1. El registro de la conversación que se usó para esta sección empieza después de este avance, así que no hay prompts de aquí que se puedan citar. Copiar el prompt exacto, qué salió de la IA, qué se cambió a mano y el análisis crítico, con el mismo formato de abajo. En el espacio de trabajo quedaron `01_dataset_pipeline.py` y `01_dataset_pipeline_v2.py`; si salieron de una conversación con IA de Annie, citarlos aquí.]

### Avance 2 (semana 9) — CBAM con compuerta [COMPLETAR]

[COMPLETAR el prompt exacto con el que se pidió el módulo de atención. En el espacio de trabajo quedaron `cbam_modulos.py` y `cbam_experimentos.ipynb`; en el repositorio, `src/cbam_modulos.py` y `notebooks/cbam.ipynb`.]

Lo que sí consta del uso posterior: la compuerta `CompuertaResidualCBAM` (y = x + γ·(CBAM(x) − x), con γ iniciado en 0 y aplicada solo a f4, compartida por detección y segmentación) se explicó y se revisó críticamente en el Avance 3 (entrada A3.1).

---

### Avance 3 (semana 10) — segmentación de instancias, SAM y salida 3D

#### A3.1 Secciones 22 y 23 y revisión crítica de la cabeza de segmentación

**Prompts (textuales):**
1. "No pero explicame detalladamente la seccion 22 23 del notebook pengwin, funcion por funcion, salidas, etc"
2. "Por que utilizamos un gate espacial y tambien un cbam"
3. "como estamos agregando la atencion en la segmentación? el dice que solo la estamos usando en el backbone"
4. "bueno a ver, yo necesito que revisemos super bien la cabeza de segmentacion, porque este es un algoritmo de segmentación "estandard", pero que tanto estamos contribuyendo a lo que estamos haciendo con la arquitectura que estamos manejando?"
5. "No, por ahora enfoquemonos en mi parte que es integrar la segmentación despues de las cajas filtradas y preparar SAM como metodo de comparación con un protocolo común. entonces, revisemos lo que llevamos hecho en este notebook y dime como se esta implementando la cabeza de segmentación" (adjuntó `proyecto_pengwin.ipynb`)

**Qué salió de la IA:** explicaciones función por función de `measure_fragment_separations` (sección 22) y de la sección 23 (objetivos semánticos, centros gaussianos y offsets; `SegmentationConvBlock`, `SkipFusionBlock`, `CenterOffsetSkipDecoder`; pérdidas; `decode_center_offset_instances`; métricas de 23.6b); la justificación de usar compuerta espacial y CBAM; y un análisis de qué aporta la cabeza frente a una segmentación estándar: es un decoder tipo U-Net con salidas de centros y offsets (estilo CenterNet y Panoptic-DeepLab), condicionado por priors de región construidos con las cajas de nuestra propia cabeza de detección y NMS.

**Qué verificó la IA y qué no:** leyó el código y las salidas guardadas del notebook. No ejecutó el entrenamiento ni midió el aporte de cada pieza: el análisis es de lectura de código, no una ablación.

**Cambios hechos a mano:** Annie definió el alcance de su parte (integrar la segmentación después de las cajas filtradas y preparar SAM con un protocolo común). [CONFIRMAR qué parte de las explicaciones se usó en el informe o en la defensa.]

**Análisis crítico:** errores de la IA detectados en la conversación:
- Leyó primero un notebook equivocado (102 celdas, sin skip connections) y se corrigió al usar el de 127 celdas que Annie indicó.
- Afirmó que no existía Dice ni IoU del modelo entrenado; era falso: el log de entrenamiento de Miguel ya guardaba el Dice de validación por región. Se corrigió.
- Afirmó que las distancias de referencia eran casi todas 0 mm; era falso: la mediana es 0,80 mm (mínimo 0,625; máximo 73,1; 575 filas, 300 principales y 275 conminutos). Se corrigió.
- Limitación: las afirmaciones sobre el aporte de la cabeza no están respaldadas por un experimento controlado.

#### A3.2 Skip connections y arquitectura tipo U-Net

**Prompts (textuales):**
1. "Por que se usa en el modelo el segmentador sin skip connection?"
2. "pero es que nosotros necesitamos entrenar con skips, o sea, que notebook te paso para que revises o cambies la parte donde no se usa el skip connection?"
3. "Tengo una pregunta, la arquitectura de segmentación que estamos haciendo es tipo UNET?"
4. "este es el best_c_conjunto" (adjuntó `best_C_conjunto.pth.zip`)

**Qué salió de la IA:** la explicación de que el decoder del entrenamiento anterior usaba solo f4 más los priors (variante tipo FCN); la indicación del notebook de entrenamiento que había que cambiar y una versión con el interruptor `USE_SKIPS`; la respuesta de que la arquitectura es tipo U-Net (skips f3, f2, f1 y una rama stem a resolución completa) con cabezas de centros y offsets, no una U-Net estándar; y la inspección del checkpoint real.

**Qué verificó la IA:** cargó `best_C_conjunto.pth` con carga estricta: decoder con skips, etapa C, época 7, y `val_seg_dice_pred` = 0,9478 guardado en el propio checkpoint. No lo reentrenó.

**Cambios hechos a mano:** el entrenamiento con skips lo hizo el equipo en su GPU (no la IA). [CONFIRMAR quién entrenó y cuándo.]

**Análisis crítico:** limitación: no hay una ablación controlada con y sin skips (mismas épocas y semilla). Comparar contra el checkpoint antiguo sin skips no la reemplaza, porque cambiaron otras cosas a la vez. Esa ablación queda pendiente del equipo.

#### A3.3 SAM como línea base zero-shot (`notebooks/sam_baseline.ipynb`)

**Prompts (textuales):**
1. "Ayudame con la parte de sam" (con un fragmento de un mensaje anterior de la IA adjunto, sobre que SAM era lo más atrasado)
2. "Como asi? o sea que primero tengo que hacer salida 3D para despues correr sam?" (con un fragmento adjunto)
3. "pero, en el proyecto, como se define que deba hacerse? que SAM coja las cajas de salida_3d o que genere sus propias cajas?"
4. "Este es el pipeline_pengwin" (adjuntó `pipeline_pengwin.py`)
5. "por que esta primera celda del ultimo sam que me diste se esta demorando mas de 20 min" (adjuntó el código de la celda)
6. "Aqui estas rutas si estan bien?" (pegó la celda de configuración, después de mostrar la estructura de carpetas reorganizada)
7. "me salio este error: esta es la ruta donde esta ese: C:\Users\annie\proyecto_analitica_corte2\resultados\modelos\entrenamiento_multitarea\best_C_conjunto.pth" (adjuntó el error `FileNotFoundError`)
8. "Por que me sale este error si supuestamente ya esta instalado?" (pegó el `ModuleNotFoundError: torchvision`)
9. "este notebook usa gpu?"
10. "esta celda en sam_baseline:" (pegó el error de la celda de configuración)
11. "Por que la carpeta modelos_sam no se puede subir al repositorio?"
12. "esta fue la salida, dame la interpretacion de esos resultados, por que el DICE es 0?" (adjuntó la galería de peores cortes y la tabla de resultados)
13. "Pero entonces el SAM si funciono? o no?"
14. "MAX_SLICES_PER_CASE = None" (confirmando que corrió sin subconjunto de cortes)
15. "como hago esto, como congelo el modelo, o es solo correrlo con ese cabio?"
16. "si por favor" (aceptando que la galería se filtrara por tamaño de referencia)

**Qué salió de la IA:** `sam_baseline.ipynb`, un notebook autónomo que:
- carga el checkpoint del modelo propio (con o sin skips) y predice las cajas con la cabeza grid y el NMS propio, con la regla de `CONFIG_PIPELINE`;
- usa **esas mismas cajas** como prompt de SAM `vit_b` y como priors del decoder propio, sobre los mismos cortes, la misma imagen 256×256 y la misma métrica (Dice e IoU por región, intersecciones acumuladas);
- guarda el protocolo en JSON, las máscaras de SAM por caso, un CSV de Dice por corte y una galería de peores cortes.

La decisión sobre de dónde salen las cajas (de la propia cabeza y no de `salida_3d`) se tomó leyendo la consigna (4.2): SAM se ejecuta sin entrenar y con el bounding box predicho como prompt. También salieron las respuestas sobre por qué `modelos_sam/` no se sube a Git (unos 375 MB; GitHub bloquea archivos de más de 100 MB), cómo "congelar" el modelo antes de correr `test` (un acuerdo del equipo sobre qué no se vuelve a tocar), y la interpretación de los resultados.

**Resultado real (val, 5 pacientes, todos los cortes con hueso, CPU):** Dice medio 0,795 y IoU medio 0,661 para SAM frente a Dice 0,948 e IoU 0,902 para el modelo propio con las mismas cajas. La cifra del modelo propio coincide con el `val_seg_dice_pred` del checkpoint (0,9478), lo que valida el protocolo. La comparación es a nivel de región, no de fragmento. `test` está pendiente. [COMPLETAR con la latencia de SAM por corte y el dispositivo que imprimió la corrida.]

**Qué verificó la IA y qué no:** probó el flujo con un SAM simulado y con pesos sintéticos y reales del modelo propio, y comprobó que las cajas y la segmentación del notebook equivalen a `inferir_cortes` con pesos aleatorios. No ejecutó SAM real ni los datos reales de Annie: no tenía el checkpoint de SAM ni los volúmenes.

**Cambios hechos a mano:** Annie instaló `segment-anything` y `torchvision` en su `.venv`, descargó `sam_vit_b_01ec64.pth` a `modelos_sam/`, ajustó las rutas a la estructura reorganizada por el equipo, copió y pegó los cambios en las celdas, y ejecutó la corrida en `val`. [CONFIRMAR dispositivo y duración de la corrida.]

**Análisis crítico:** errores de la IA:
- La primera versión dependía de los resultados de `salida_3d`; se rehízo autónoma.
- Una prueba de equivalencia con el pipeline real detectó una diferencia (argmax de logits en torch frente a argmax de softmax en numpy); se corrigió en la réplica de la inferencia.
- Mostraba el error genérico "SAM es opcional" y ocultaba que lo que faltaba era `torchvision`; se cambió para mostrar el error real.
- Asumió las rutas del repositorio anterior (`proyecto_pengwin.ipynb`, `evidencias/entrenamiento_multitarea/`) y fallaron tras la reorganización; hubo varias rondas de corrección. Una de ellas fue un consejo equivocado (copiar `cbam_modulos.py` a la raíz) cuando el archivo estaba en `src/` y lo que faltaba era agregar `src/` al `sys.path`.
- Atribuyó los 20 minutos de la primera celda al entorno sin poder reproducirlos en la máquina de Annie. [CONFIRMAR si se resolvió.]
- La galería de "peores cortes" mostraba cortes con unos pocos píxeles de referencia, con Dice 0 que no representan a SAM; se filtró por tamaño de referencia (`MIN_GT_PIXELS`) y se añadió una tabla que separa referencias pequeñas y grandes.

Limitaciones: resultados solo en `val` (5 casos), con un modelo cuya época se eligió en esos mismos casos; SAM en `test` pendiente; corrida en CPU, así que no es la medición formal de latencia.

#### A3.4 Salida 3D (`notebooks/salida_3d.ipynb`) y adaptación al repositorio reorganizado

**Prompts (textuales):**
1. "esta celda de salidas_3D como le cambio la ruta:" (adjuntó el código de la celda y el error por `cbam_modulos`)
2. "no porque cbam_modulos.py esta aqui: C:\Users\annie\proyecto_analitica_corte2\src\cbam_modulos.py"
3. "todavia me sale el mismo error, esta es la priemra celda" (adjuntó la celda)
4. "Por que depende de proyecto pengwin si ese ya no existe, como te mostre, mi compañero reorganizo el proyecto" (con el listado de la raíz y, después, de `notebooks/` y `src/`)
5. Pegó el contenido de `src/definiciones_notebooks.py` y `src/rutas.py`.
6. "Bien, ahora, teniendo en cuenta que ya se hizo lo del sam y lo de la salida 3C, necesito que revises los notebooks, me digas si hay que hacer algún cambio y que me  digas, de mi parte Annie, que me falta hacer para esta entrega" (adjuntó `sam_baseline.ipynb`, `Segmentacion_instancias.ipynb`, `salida_3d.ipynb`, `ajuste_asociacion_mascaras_reales_val.csv` y una figura de la curva de asociación)
7. "estoy corriendo el Segmentación instancias, me sale este error en la primera celda:" (pegó el `StopIteration`)
8. "Las ultimas celdas, de comparacion cualitativa con sam usando cajas predichas, y prueba de contrato con sam con predictor simulado, es necesario tenerlo? eso no es lo que se hace en el notebook de sam?"

La primera versión de `salida_3d.ipynb` (9 de octubre) también salió de una conversación con IA. [COMPLETAR el prompt exacto con el que se pidió.]

**Qué salió de la IA:** la adaptación de la celda 1 de `salida_3d.ipynb` para usar `rutas.py` y `definiciones_notebooks.py` en lugar de `proyecto_pengwin.ipynb`, la corrección de la ruta del modelo (`resultados/modelos/entrenamiento_multitarea/`), la carpeta de caché y la de métricas; la corrección de `Segmentacion_instancias.ipynb` (quitar `mode="I"` de Pillow en `restore_letterbox_label_mask`, actualizar el texto de las secciones 9 y 13); y la lectura crítica de resultados: el F1 de 1,000 de la asociación 3D con máscaras reales (celdas 10.1 y 10.2) es un techo del armado, ajustado en los mismos 5 casos de `val`, y no es el resultado del modelo. El notebook de `salida_3d` se saltaba en silencio las celdas del modelo (10.3 a 10.9) porque no encontraba los pesos, y solo se notó al revisar las salidas guardadas.

**Qué verificó la IA y qué no:** probó `salida_3d.ipynb` adaptado en una copia con la misma estructura de carpetas y con el checkpoint real exportado, hasta la celda 10.3 (carga del modelo e inferencia de los cortes). No ejecutó 10.4 en adelante (ajuste con el modelo, distancias, archivos de salida), así que los resultados con el modelo siguen pendientes de correrse.

**Cambios hechos a mano:** Annie reemplazó los notebooks por las versiones corregidas, ejecutó la celda 1 en su máquina y aportó el listado de carpetas y los archivos del equipo que la IA no tenía. [CONFIRMAR qué se ejecutó de 10.3 en adelante.]

**Análisis crítico:** errores de la IA en esta parte: asumió una estructura de repositorio que ya no existía (ver A3.3); tardó en identificar que la celda 1 editada había quedado sin los imports de `deteccion_grid`, lo que repitió el `NameError` por `NMS_CONFIG`; y diseñó el notebook para saltarse en silencio las celdas del modelo, sin detenerse cuando no encontraba los pesos. Limitaciones: el ajuste usa 5 casos de `val`; `min_voxels=50` salió de ese ajuste, y como en `val` no hay fragmentos reales de menos de 100 vóxeles, hay que validarlo con los tamaños reales de `train` antes de congelarlo. La sobresegmentación con el modelo (caso 035) no está resuelta.

#### A3.5 Revisión de pendientes del avance

**Prompts (textuales):**
1. "no pero creo que yo no he hecho lo de SAM, o sea revisemos toda la entrega y dime que me falta"
2. "Esto es lo que el claude de mi compañero fabian le dio y le dijo que faltaba, teneindo en cuenta lo que hemos hecho, dime que partes verdaderamente faltan y cuales ya cumplimos" (adjuntó el PDF "Estado del proyecto PENGWIN — rumbo a la entrega final")
3. "Pregunta, a mi me tocaba hacer lo de los fragmentos en algun avance?"

**Qué salió de la IA:** una lista de pendientes de Annie y del equipo contrastada con la consigna, y una separación entre lo ya cumplido (SAM real en `val`, adaptación de los notebooks), lo hecho a medias (salida 3D con el modelo, Dice por fragmento) y lo que falta (SAM en `test`, visualizadores 2 y 3, esta sección).

**Análisis crítico:** la IA no tiene el reparto interno del equipo; sobre qué le tocaba a Annie se apoyó en el documento de otro integrante y lo dejó marcado como sugerencia. Aquí el criterio lo pone el equipo, no la IA.
