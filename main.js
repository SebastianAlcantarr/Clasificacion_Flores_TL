import * as tf from '@tensorflow/tfjs';
import { cargarKeras3 } from './keras3.js';

const $ = (id) => document.getElementById(id);
const BASE = import.meta.env.BASE_URL;

// Índice del modelo = orden alfabético de las carpetas de entrenamiento ('0','1','10','100',...)
const NOMBRES = [
  'pink ssprimrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily',
  'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea',
  'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
  'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
  'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation',
  'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
  'great masterwort', 'siam tulip', 'lenten rose', 'barberton daisy', 'daffodil', 'sword lily', 'poinsettia',
  'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common dandelion', 'petunia', 'wild pansy',
  'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
  'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush',
  'californian poppy', 'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy', 'gazania', 'azalea',
  'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium',
  'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen',
  'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia',
  'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily', 'common tulip',
  'wild rose',
];
const CLASES = Array.from({ length: 104 }, (_, i) => String(i)).sort().map((id) => NOMBRES[Number(id)]);
const TAM = 160;
// Si la clase ganadora no supera este umbral, no se afirma nada
const UMBRAL = 0.4;

// ---------- Modelos ----------
const ETIQUETAS = { mobilenet: 'MobileNetV2', efficientnet: 'EfficientNetB0' };
const modelos = { mobilenet: null, efficientnet: null };
const RUTAS = {
  mobilenet: `${BASE}web_flores_mobilenet/model.json`,
  efficientnet: `${BASE}web_flores_efficientB0/model.json`,
};

const cargas = {};
const barra = $('carga');

function mostrarCarga(clave, p) {
  const pct = Math.round(p * 100);
  barra.hidden = false;
  $('carga-texto').textContent = `Cargando ${ETIQUETAS[clave]}…`;
  $('carga-pct').textContent = `${pct}%`;
  $('carga-barra').style.width = `${pct}%`;
  $('carga-barra').parentElement.setAttribute('aria-valuenow', pct);
}

// Solo se descarga el modelo en uso (el otro, al cambiar): ahorra datos y memoria en el teléfono
function cargarModelo(clave) {
  if (modelos[clave]) return Promise.resolve();
  if (!cargas[clave]) {
    cargas[clave] = cargarKeras3(RUTAS[clave], (p) => activo === clave && mostrarCarga(clave, p))
      .then((m) => {
        modelos[clave] = m;
        $('estado-modelos').hidden = true;
        if (activo === clave) barra.hidden = true;
        if (activo === clave && !stream && !foto.hidden && foto.src) predecir(foto, foto.naturalWidth, foto.naturalHeight);
      })
      .catch((err) => {
        console.error(err);
        cargas[clave] = null;
        barra.hidden = true;
        $('estado-modelos').textContent = `No se pudo cargar ${ETIQUETAS[clave]}. Revisa tu conexión y toca «Reintentar».`;
        $('estado-modelos').hidden = false;
        $('reintentar').hidden = false;
      });
  }
  return cargas[clave];
}

// ---------- Modelo activo ----------
let activo = 'mobilenet';

$('reintentar').addEventListener('click', () => {
  $('reintentar').hidden = true;
  $('estado-modelos').hidden = true;
  cargarModelo(activo);
});

$('cambiar-modelo').addEventListener('click', () => {
  activo = activo === 'mobilenet' ? 'efficientnet' : 'mobilenet';
  $('cambiar-modelo').textContent = `Modelo: ${ETIQUETAS[activo]}`;
  Object.keys(ETIQUETAS).forEach((k) => ($(`win-${k}`).hidden = k !== activo));
  barra.hidden = true;
  $('reintentar').hidden = true;
  if (modelos[activo]) {
    // libera el modelo anterior para no agotar la memoria del móvil
    const otro = activo === 'mobilenet' ? 'efficientnet' : 'mobilenet';
    modelos[otro]?.liberar();
    modelos[otro] = null;
    cargas[otro] = null;
  } else {
    cargarModelo(activo);
  }
  if (!stream && foto.src && !foto.hidden) predecir(foto, foto.naturalWidth, foto.naturalHeight);
});

cargarModelo(activo);

// ---------- Cámara ----------
const video = $('video');
const foto = $('foto');
const entrada = $('entrada');
const ctxEntrada = entrada.getContext('2d', { willReadFrequently: true });
ctxEntrada.imageSmoothingQuality = 'high';

let stream = null;
let facingMode = 'environment';
let bucle = null;
let ocupado = false;

async function iniciarCamara() {
  $('camara-error').textContent = '';
  if (!navigator.mediaDevices?.getUserMedia) {
    $('camara-error').textContent = 'Tu navegador no permite usar la cámara (se requiere HTTPS).';
    return;
  }
  detenerCamara();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode, width: { ideal: 640 }, height: { ideal: 640 } },
    });
  } catch (err) {
    console.error(err);
    $('camara-error').textContent = 'No se pudo acceder a la cámara. Revisa los permisos del navegador.';
    return;
  }
  video.srcObject = stream;
  video.classList.toggle('-scale-x-100', facingMode === 'user');
  video.hidden = false;
  foto.hidden = true;
  $('stage-vacio').hidden = true;
  $('camara').textContent = 'Detener cámara';
  $('cambiar-camara').disabled = false;
  await video.play().catch(() => {});
  buclePrediccion();
}

function detenerCamara() {
  clearTimeout(bucle);
  bucle = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  $('camara').textContent = 'Activar cámara';
  $('cambiar-camara').disabled = true;
  if (foto.hidden) $('stage-vacio').hidden = false;
}

function buclePrediccion() {
  if (!stream) return;
  if (video.readyState >= 2) predecir(video, video.videoWidth, video.videoHeight);
  bucle = setTimeout(buclePrediccion, 300);
}

// ---------- Predicción ----------
function top3(probs) {
  return Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
}

function pintar(clave, probs) {
  const [[mejor, p], ...resto] = top3(probs);
  const seguro = p >= UMBRAL;
  const nombre = $(`nombre-${clave}`);
  nombre.textContent = seguro ? CLASES[mejor] : `¿${CLASES[mejor]}?`;
  nombre.classList.toggle('apagado', !seguro);
  $(`conf-${clave}`).textContent = `${(p * 100).toFixed(1)}% segura`;
  $(`otros-${clave}`).textContent = resto.map(([i, q]) => `${CLASES[i]} ${(q * 100).toFixed(0)}%`).join(' · ');
}

function predecir(fuente, w, h) {
  if (!w || !h || ocupado || !modelos[activo]) return;
  ocupado = true;
  try {
    const lado = Math.min(w, h);
    ctxEntrada.drawImage(fuente, (w - lado) / 2, (h - lado) / 2, lado, lado, 0, 0, TAM, TAM);
    // Ambos modelos llevan el preprocesado dentro: reciben RGB 0-255
    const probs = tf.tidy(() => modelos[activo].predict(tf.browser.fromPixels(entrada).toFloat().expandDims(0)).dataSync());
    pintar(activo, probs);
  } catch (err) {
    console.error(err);
    $('estado-modelos').textContent = `Error al predecir: ${err.message}`;
    $('estado-modelos').hidden = false;
  } finally {
    ocupado = false;
  }
}

$('camara').addEventListener('click', () => (stream ? detenerCamara() : iniciarCamara()));

$('cambiar-camara').addEventListener('click', () => {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  iniciarCamara();
});

$('archivo').addEventListener('change', (e) => {
  const archivo = e.target.files[0];
  if (!archivo) return;
  detenerCamara();
  if (foto.src) URL.revokeObjectURL(foto.src);
  foto.onload = () => {
    foto.hidden = false;
    video.hidden = true;
    $('stage-vacio').hidden = true;
    predecir(foto, foto.naturalWidth, foto.naturalHeight);
  };
  foto.src = URL.createObjectURL(archivo);
  e.target.value = '';
});
