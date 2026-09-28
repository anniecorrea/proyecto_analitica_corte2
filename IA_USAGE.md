# Registro de uso de IA

## Semana 9 - Datos y targets de deteccion

- Integrante responsable: Daniela.
- Herramienta: OpenAI Codex.
- Prompt del usuario: "bueno bueno amiguis, tenemos hasta las 2pm para realizar la parte mia que es la de datos targets".
- Uso: apoyo para estructurar el cargador 2D, transformar las mascaras `.mha`
  en clases anatomicas, obtener bounding boxes por region, preparar el dataset
  compatible con PyTorch, crear pruebas y generar evidencia visual.
- Decision revisada: los IDs 1-10 se asignan a sacro, 11-20 a coxal
  izquierdo y 21-30 a coxal derecho. Los fragmentos de cada region se unen
  en una caja porque el requisito de deteccion pide localizar regiones
  anatomicas; las instancias se conservan para la segmentacion posterior.
- Validacion realizada: cuatro pruebas unitarias, lectura real de los splits,
  prueba de integracion con PyTorch y comprobacion de que no existen pacientes
  compartidos entre train, validacion y test.
- Ajuste manual/critico: el umbral minimo de pixeles se escala despues del
  letterbox para evitar perder regiones pequenas al redimensionar a 256 x 256.
- Limitaciones: este modulo prepara ground truth; no implementa ni evalua el
  backbone, CBAM, la cabeza de deteccion ni NMS. El equipo debe revisar el
  contrato de targets al integrarlo con la arquitectura final.
