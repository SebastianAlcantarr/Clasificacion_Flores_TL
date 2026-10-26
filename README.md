# Clasificación con redes neuronales

Una sola página web con dos modelos de TensorFlow.js que se ejecutan 100% en el navegador:

- **Números**: dibuja un dígito del 0 al 9 y la red lo clasifica (entrada 28×28, escala de grises).
- **Perros y gatos**: clasificación en tiempo real con la cámara o subiendo una imagen (entrada 150×150 a color).

Basado en el material del canal de YouTube [Ringa Tech](https://youtube.com/RingaTech).

## Estructura

```
index.html                    Página única (pestañas Números / Perros y gatos)
main.js                       Lógica: dibujo, cámara y predicciones
style.css                     Estilos (Tailwind CSS v4)
public/models/numeros/        Modelo de números
public/models/perros-gatos/   Modelo de perros y gatos
```

## Desarrollo

```bash
npm install
npm run dev
```

## Subir a un servidor

```bash
npm install
npm run build
```

Sube **todo el contenido de la carpeta `dist/`** a tu servidor (Apache, Nginx, Netlify, Vercel, GitHub Pages, hosting compartido, etc.).
Las rutas son relativas, así que funciona tanto en la raíz del dominio como en una subcarpeta.

Para probar el build localmente: `python -m http.server 8000 -d dist` y abre http://localhost:8000

> **Importante:** la cámara solo funciona con **HTTPS** (o en `localhost`). En un servidor sin HTTPS
> el clasificador de números y la opción "Subir una imagen" siguen funcionando.
