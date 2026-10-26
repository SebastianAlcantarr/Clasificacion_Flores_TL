# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Visitantes de un portafolio público: cualquiera que llegue al enlace, desde laptop o celular, con curiosidad por probar dos redes neuronales directamente en el navegador.

## Product Purpose
Demostración interactiva de dos modelos de TensorFlow.js que corren en el navegador: reconocimiento de dígitos escritos a mano (MNIST, entrada 28×28) y clasificación perro/gato (entrada 150×150, cámara o foto). El éxito es que el visitante entienda cómo "ve" la red: qué entrada recibe realmente y cómo reparte su confianza entre las clases.

## Positioning
Todo ocurre en el dispositivo del visitante, sin servidor: la página puede mostrar la entrada exacta que recibe el modelo (el dibujo reducido a 28×28, el recorte de 150×150) junto a la salida, en vivo.

## Capabilities and Constraints
- Dígitos: lienzo de dibujo, predicción automática al terminar cada trazo o manual, vista previa 28×28, probabilidad por dígito 0–9.
- Perros y gatos: cámara (frontal/trasera, en bucle cada 150 ms) o foto subida; recorte cuadrado centrado a 150×150; probabilidad gato vs. perro.
- Modelos en `public/models/numeros` y `public/models/perros-gatos`; Vite + Tailwind v4.
- La cámara requiere HTTPS.

## Brand Commitments
- Título: "Tarea de Redes Neuronales", sin cambios.
- Todo en español.
- Fuentes: Space Grotesk y JetBrains Mono.

## Evidence on Hand
Solo los dos modelos entrenados. No hay métricas de precisión, datasets ni capturas documentadas; no inventarlas.

## Product Principles
- Mostrar, no decir: la entrada real del modelo siempre visible junto a su respuesta.
- La incertidumbre es contenido: las probabilidades importan tanto como la clase ganadora.
- Probar en segundos, en cualquier pantalla.
