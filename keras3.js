// Ejecuta modelos exportados desde Keras 3 (model.json + pesos), que tfjs-layers no sabe leer.
import * as tf from '@tensorflow/tfjs';

const ref = (a) => a?.class_name === '__keras_tensor__' ? a.config.keras_history[0] : null;

function activar(x, nombre) {
  switch (nombre) {
    case 'linear': return x;
    case 'relu': return tf.relu(x);
    case 'silu': case 'swish': return tf.mul(x, tf.sigmoid(x));
    case 'sigmoid': return tf.sigmoid(x);
    case 'softmax': return tf.softmax(x);
    default: throw new Error(`Activación no soportada: ${nombre}`);
  }
}

function capa(l, entradas, args, w) {
  const c = l.config;
  const n = c.name;
  const x = Array.isArray(entradas[0]) ? entradas[0] : entradas[0];
  const peso = (k) => w[`${n}/${k}`];
  switch (l.class_name) {
    case 'Conv2D': {
      let y = tf.conv2d(x, peso('kernel'), c.strides, c.padding, 'NHWC', c.dilation_rate);
      if (c.use_bias) y = tf.add(y, peso('bias'));
      return activar(y, c.activation);
    }
    case 'DepthwiseConv2D': {
      const k = w[`${n}/depthwise_kernel`] ?? peso('kernel');
      let y = tf.depthwiseConv2d(x, k, c.strides, c.padding, 'NHWC', c.dilation_rate);
      if (c.use_bias) y = tf.add(y, peso('bias'));
      return activar(y, c.activation);
    }
    case 'BatchNormalization':
      return tf.batchNorm(x, peso('moving_mean'), peso('moving_variance'),
        c.center ? peso('beta') : undefined, c.scale ? peso('gamma') : undefined, c.epsilon);
    case 'ReLU': return c.max_value != null ? tf.clipByValue(x, 0, c.max_value) : tf.relu(x);
    case 'Activation': return activar(x, c.activation);
    case 'ZeroPadding2D': return tf.pad(x, [[0, 0], ...c.padding, [0, 0]]);
    case 'Add': return entradas[0].reduce((a, b) => tf.add(a, b));
    case 'Multiply': return entradas[0].reduce((a, b) => tf.mul(a, b));
    case 'GlobalAveragePooling2D': return tf.mean(x, [1, 2], c.keepdims);
    case 'Reshape': return tf.reshape(x, [-1, ...c.target_shape]);
    case 'Dropout': return x;
    case 'Rescaling': return tf.add(tf.mul(x, c.scale), c.offset);
    case 'Normalization': {
      const m = peso('mean');
      const v = peso('variance');
      return tf.div(tf.sub(x, m), tf.maximum(tf.sqrt(v), 1e-7));
    }
    case 'Dense': return activar(tf.add(tf.matMul(x, peso('kernel')), peso('bias')), c.activation);
    case 'TrueDivide': return tf.div(x, args[1]);
    case 'Subtract': return tf.sub(x, args[1]);
    default: throw new Error(`Capa no soportada: ${l.class_name}`);
  }
}

function correr(cfg, entrada, w) {
  const env = {};
  const [[nombreIn]] = cfg.input_layers.length && Array.isArray(cfg.input_layers[0]) ? cfg.input_layers : [cfg.input_layers];
  for (const l of cfg.layers) {
    if (l.class_name === 'InputLayer') {
      env[l.name] = entrada;
      continue;
    }
    const args = l.inbound_nodes[0].args;
    const entradas = args.map((a) => (Array.isArray(a) ? a.map((t) => env[ref(t)]) : ref(a) ? env[ref(a)] : a));
    env[l.name] = l.class_name === 'Functional'
      ? correr(l.config, entradas[0], w)
      : capa(l, entradas, args, w);
  }
  return env[cfg.output_layers[0] === undefined ? '' : (Array.isArray(cfg.output_layers[0]) ? cfg.output_layers[0][0] : cfg.output_layers[0])];
}

// Descarga con reintentos y progreso (0-1). Secuencial para no saturar la memoria ni la red del teléfono.
async function bajar(url, tipo, alBytes, intentos = 3) {
  for (let i = 1; ; i++) {
    let previo = 0;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status} en ${url}`);
      const largo = Number(r.headers.get('content-length')) || 0;
      if (!r.body) {
        const b = tipo === 'json' ? await r.json() : await r.arrayBuffer();
        alBytes(largo || 1, largo || 1);
        return b;
      }
      const trozos = [];
      let recibidos = 0;
      const lector = r.body.getReader();
      for (;;) {
        const { done, value } = await lector.read();
        if (done) break;
        trozos.push(value);
        recibidos += value.length;
        alBytes(recibidos - previo, largo);
        previo = recibidos;
      }
      const buf = new Uint8Array(recibidos);
      let o = 0;
      trozos.forEach((t) => { buf.set(t, o); o += t.length; });
      return tipo === 'json' ? JSON.parse(new TextDecoder().decode(buf)) : buf.buffer;
    } catch (err) {
      alBytes(-previo, 0);
      if (i >= intentos) throw err;
      await new Promise((res) => setTimeout(res, 800 * i));
    }
  }
}

export async function cargarKeras3(ruta, alProgreso = () => {}) {
  await tf.ready();
  const base = ruta.slice(0, ruta.lastIndexOf('/') + 1);
  let hecho = 0;
  // El tamaño total se conoce al terminar el model.json (~ pesos ocupan 4 MB por shard); se estima y se corrige.
  let estimado = 1;
  const avance = (d) => {
    hecho = Math.max(0, hecho + d);
    alProgreso(Math.min(0.99, hecho / estimado));
  };
  alProgreso(0);
  const mj = await bajar(ruta, 'json', (d) => { hecho = 0; void d; });
  const grupo = mj.weightsManifest[0];
  const tam = grupo.weights.reduce((s, p) => s + p.shape.reduce((a, b) => a * b, 1) * (p.quantization ? 1 : 4), 0);
  estimado = tam || grupo.paths.length * 4194304;
  const partes = [];
  for (const p of grupo.paths) partes.push(await bajar(base + p, 'bin', avance));
  const total = new Uint8Array(partes.reduce((s, b) => s + b.byteLength, 0));
  let off = 0;
  partes.forEach((b) => { total.set(new Uint8Array(b), off); off += b.byteLength; });
  partes.length = 0;
  const w = tf.io.decodeWeights(total.buffer, grupo.weights);
  const cfg = mj.modelTopology.model_config.config;
  alProgreso(1);
  return {
    predict: (x) => tf.tidy(() => correr(cfg, x, w)),
    liberar: () => Object.values(w).forEach((t) => t.dispose()),
  };
}
