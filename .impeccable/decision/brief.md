# Surface: index.html (demo de las dos redes)

Mode: Experience. Visitante de portafolio público, laptop o celular. Tarea: dibujar un dígito o mostrar un perro/gato y entender qué ve la red. Restricciones: título "Tarea de Redes Neuronales", español, Space Grotesk + JetBrains Mono, Tailwind v4.

## Direction contract

THESIS: La red ve en píxeles; la página también. Un escritorio de 1 bit donde la entrada real del modelo es una ventana más, ampliada a píxel entero. Rechaza la demo oscura de dos tarjetas con barras de degradado.

OWN-WORLD: Solo #000 y #fff. Todo gris es tramado ordenado (25/50/75%). Ventanas con filete negro de 2px, barra de título negra con nombre en mayúsculas Space Grotesk, caja de cierre, barra de estado en JetBrains Mono, sombra sólida desplazada. Botones blancos con borde doble para el predeterminado; presionar invierte. Selección con hormigas marchantes. Traducción: la fuente de píxel del mundo se sustituye por las fuentes fijadas.

STORY: El visitante dibuja, ve su trazo reducido a 28×28 en la ventana «Lo que ve la red» y la probabilidad de cada dígito; con la cámara ve el recorte 150×150. Entiende que la red decide sobre esos píxeles.

FIRST VIEWPORT: Barra de menú arriba (título, pestañas Números / Perros y gatos como elementos de menú, estado de modelos). Debajo, sobre tramado 50%: ventana Lienzo (izquierda, grande, con Limpiar/Predecir), ventana «Lo que ve la red · 28×28» (centro, ampliada pixelada con rejilla), ventana Predicción (dígito gigante y confianza) y Probabilidades (barras tramadas, ganadora sólida).

FORM: Escritorio de 1 bit (challenger medium-native-one-bit-desktop, elegido por el usuario). seed: a9831aac. Firma: la ventana de entrada del modelo con hormigas marchantes mientras la red está leyendo.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
