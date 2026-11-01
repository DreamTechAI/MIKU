import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const wrap = $('#canvas-wrap');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
} catch (error) {
  $('#loading').hidden = true; $('#error').hidden = false;
  $('#error-message').textContent = '当前浏览器无法创建 WebGL 2 上下文。请开启硬件加速，或用支持 WebGL 2 的 Chrome / Edge / Safari 打开。';
  throw error;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
wrap.appendChild(renderer.domElement);
renderer.domElement.setAttribute('aria-label', '拖动旋转模型，滚轮缩放，右键平移');
renderer.domElement.setAttribute('role', 'img');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#090c0d');
scene.fog = new THREE.FogExp2('#090c0d', 0.015);
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 150);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.065;
controls.minDistance = 1.6; controls.maxDistance = 16;
controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotateSpeed = 0.65;
const turntable = new THREE.Group(); scene.add(turntable);
const orientation = new THREE.Group(); turntable.add(orientation);
let asset = null, assetName = 'Dragonbone Axe', originalMaterials = new Map(), temporaryMaterials = [];
let triangles = 0, requestId = 0, channel = 'beauty', activePreset = 'noir', toastTimer, activeFetch = null;
let desiredRotation = 0, currentRotation = 0, pendingCapture = false;
const environments = new Map();
let studioEnvironment = null, studioHDR = null;
let photo = null, photoMode = false, photoBuilding = false, photoToken = 0, photoExport = null;
const pmrem = new THREE.PMREMGenerator(renderer);
pmrem.compileCubemapShader();

const presets = {
  noir: { bg: '#090c0d', floor: '#181c1e', exposure: 1.25, intensity: 1.8,
    panels: [[[-3, 4, 3], [3, 6], '#eaf3ff', 7], [[4, 3, -2], [2, 5], '#c4d9ff', 9], [[0, 7, 1], [5, 3], '#fff2db', 5], [[0, 3, 5], [6, 5], '#ffffff', 4.5]], key: '#fff0d8', fill: '#c4ddff' },
  soft: { bg: '#353e3d', floor: '#414b49', exposure: 1.15, intensity: 2.0,
    panels: [[[-4, 4, 4], [6, 6], '#ffffff', 5], [[4, 3, 2], [5, 5], '#dceaff', 4], [[0, 7, -3], [6, 5], '#fff4e6', 5], [[0, 2, -6], [5, 4], '#ffffff', 2]], key: '#fff8ec', fill: '#e3f0ff' },
  ember: { bg: '#130d0c', floor: '#25201b', exposure: 1.25, intensity: 1.65,
    panels: [[[-4, 3, 3], [3, 6], '#ffe1ac', 8], [[3, 4, -2], [2, 5], '#a9beff', 10], [[0, 6, 1], [5, 3], '#fff2db', 5], [[0, 3, 6], [6, 5], '#fff5ed', 4]], key: '#ffe4b2', fill: '#a5baff' },
};
function environmentFor(name) {
  if (environments.has(name)) return environments.get(name).texture;
  const env = new THREE.Scene(); env.background = new THREE.Color(name === 'soft' ? '#333b3b' : '#101415');
  for (const [position, size, tint, power] of presets[name].panels) {
    const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(tint).multiplyScalar(power), side: THREE.DoubleSide });
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(...size), material);
    panel.position.fromArray(position); panel.lookAt(0, 1, 0); env.add(panel);
  }
  const target = pmrem.fromScene(env, 0.04, 0.1, 100);
  environments.set(name, target);
  env.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
  return target.texture;
}

const lightRig = new THREE.Group(); scene.add(lightRig);
const key = new THREE.DirectionalLight('#fff0db', 2.4); key.position.set(-3, 6, 4);
key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -3.5, right: 3.5, top: 4, bottom: -3, near: 0.5, far: 20 });
key.shadow.bias = -0.00015; key.shadow.normalBias = 0.018; key.shadow.radius = 3;
lightRig.add(key);
const fill = new THREE.DirectionalLight('#b7ceff', 0.6); fill.position.set(4, 3, -3); lightRig.add(fill);
const ambient = new THREE.HemisphereLight('#b6c7d0', '#111719', 0.3); scene.add(ambient);

// A real planar reflection, softly filtered and faded into the studio floor.
const mirrorShader = {
  uniforms: { tDiffuse: { value: null }, textureMatrix: { value: new THREE.Matrix4() }, color: { value: new THREE.Color('#0c1011') }, backdrop: { value: new THREE.Color('#090c0d').convertLinearToSRGB() }, strength: { value: 0.55 }, texel: { value: new THREE.Vector2(1 / 1024, 1 / 1024) } },
  vertexShader: `uniform mat4 textureMatrix; varying vec4 vMirror; varying vec3 vLocal; varying vec3 vViewPosition;
    void main(){vLocal=position;vMirror=textureMatrix*vec4(position,1.0);vec4 mv=modelViewMatrix*vec4(position,1.0);vViewPosition=mv.xyz;gl_Position=projectionMatrix*mv;}`,
  fragmentShader: `uniform sampler2D tDiffuse;uniform vec3 color;uniform vec3 backdrop;uniform float strength;uniform vec2 texel;varying vec4 vMirror;varying vec3 vLocal;varying vec3 vViewPosition;
    void main(){vec2 uv=vMirror.xy/vMirror.w;vec2 d=texel*2.4;vec3 reflection=texture2D(tDiffuse,uv).rgb*0.20;
      reflection+=texture2D(tDiffuse,uv+vec2(d.x,0.)).rgb*.12;reflection+=texture2D(tDiffuse,uv-vec2(d.x,0.)).rgb*.12;
      reflection+=texture2D(tDiffuse,uv+vec2(0.,d.y)).rgb*.12;reflection+=texture2D(tDiffuse,uv-vec2(0.,d.y)).rgb*.12;
      reflection+=texture2D(tDiffuse,uv+d).rgb*.08;reflection+=texture2D(tDiffuse,uv-d).rgb*.08;
      reflection+=texture2D(tDiffuse,uv+vec2(d.x,-d.y)).rgb*.08;reflection+=texture2D(tDiffuse,uv+vec2(-d.x,d.y)).rgb*.08;
      float fade=exp(-length(vLocal.xy)*0.12);gl_FragColor=vec4(mix(color,reflection,strength*fade),1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      gl_FragColor.rgb=mix(gl_FragColor.rgb,backdrop,smoothstep(8.0,28.0,length(vViewPosition)));
    }`,
};
const mirror = new Reflector(new THREE.PlaneGeometry(200, 200), { textureWidth: 1024, textureHeight: 1024, clipBias: 0.003, shader: mirrorShader, multisample: 2 });
mirror.rotation.x = -Math.PI / 2; mirror.position.y = -0.012; scene.add(mirror);
const plainFloor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshBasicMaterial({ color: '#0c1011' }));
plainFloor.rotation.x = -Math.PI / 2; plainFloor.position.y = -0.013; plainFloor.visible = false; scene.add(plainFloor);
const shadowFloor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.32 }));
shadowFloor.rotation.x = -Math.PI / 2; shadowFloor.position.y = -0.007; shadowFloor.receiveShadow = true; scene.add(shadowFloor);
// A soft photographic pool of light makes the floor readable without adding a platform.
const poolCanvas = document.createElement('canvas'); poolCanvas.width = poolCanvas.height = 128;
const ctx = poolCanvas.getContext('2d'), gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
gradient.addColorStop(0, '#b7c4be2b'); gradient.addColorStop(0.4, '#9bb3a71a'); gradient.addColorStop(1, '#9bb3a700');
ctx.fillStyle = gradient; ctx.fillRect(0, 0, 128, 128);
const pool = new THREE.Mesh(new THREE.PlaneGeometry(9, 9), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(poolCanvas), transparent: true, depthWrite: false, opacity: 0.5 }));
pool.rotation.x = -Math.PI / 2; pool.position.y = -0.005; scene.add(pool);

const draco = new DRACOLoader(); draco.setDecoderPath('./vendor/draco/');
const manager = new THREE.LoadingManager();
// Imported GLBs must be self-contained; never fetch embedded external resource URLs.
manager.setURLModifier(url => {
  if (/^(blob:|data:)/i.test(url)) return url;
  throw new Error('请使用内嵌贴图的 GLB；不支持外部纹理或外部缓冲文件。');
});
const loader = new GLTFLoader(manager); loader.setDRACOLoader(draco); loader.setMeshoptDecoder(MeshoptDecoder);

function toast(message) { clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false; toastTimer = setTimeout(() => $('#toast').hidden = true, 4200); }
function updateActive(selector, value, attribute) {
  $$(selector).forEach(b => { const active = b.dataset[attribute] === value; b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active)); });
}
function updateFloor() {
  const beauty = channel === 'beauty' || channel === 'clay';
  mirror.visible = beauty && $('#reflection').checked;
  plainFloor.visible = beauty && !$('#reflection').checked;
  shadowFloor.visible = beauty && $('#shadows').checked;
  pool.visible = beauty;
  key.castShadow = beauty && $('#shadows').checked;
}
function setPreset(name) {
  activePreset = name; const p = presets[name];
  scene.environment = studioEnvironment || environmentFor(name); scene.environmentIntensity = p.intensity;
  scene.background.set(p.bg); scene.fog.color.set(p.bg);
  mirror.material.uniforms.color.value.set(p.floor); mirror.material.uniforms.backdrop.value.set(p.bg).convertLinearToSRGB(); plainFloor.material.color.set(p.floor);
  key.color.set(p.key); fill.color.set(p.fill);
  key.intensity = name === 'ember' ? 1.4 : 0.65;
  fill.intensity = name === 'ember' ? 0.7 : 0.3;
  ambient.intensity = name === 'soft' ? 0.4 : 0.2;
  $('#environment').value = p.intensity; $('#environment-value').textContent = p.intensity.toFixed(2);
  renderer.toneMappingExposure = p.exposure;
  $('#exposure').value = p.exposure; $('#exposure-value').textContent = p.exposure.toFixed(2);
  updateActive('[data-preset]', name, 'preset'); syncPhotoLighting();
}
function resetCamera() {
  const aspect = wrap.clientWidth / wrap.clientHeight;
  const d = aspect < 0.9 ? 9.3 : aspect < 1.3 ? 7.8 : 6.8;
  camera.position.set(d * 0.12, 2.05, d);
  controls.target.set(0, 1.45, 0);
  controls.update(); controls.saveState();
}
function fitModel() {
  orientation.position.set(0, 0, 0); orientation.scale.setScalar(1); orientation.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(orientation), size = box.getSize(new THREE.Vector3());
  if (!Number.isFinite(size.length()) || size.length() < 1e-9) throw new Error('模型没有可显示的有效几何体。');
  const scale = 2.9 / Math.max(size.x, size.y, size.z);
  orientation.scale.setScalar(scale); orientation.updateMatrixWorld(true);
  box.setFromObject(orientation);
  const center = box.getCenter(new THREE.Vector3());
  orientation.position.set(-center.x, -box.min.y + 0.008, -center.z);
  orientation.updateMatrixWorld(true); resetCamera();
}
function disposeAsset(object) {
  const mats = new Set(), textures = new Set(), geometries = new Set(), skeletons = new Set();
  object.traverse(o => { if (o.geometry) geometries.add(o.geometry); if (o.skeleton) skeletons.add(o.skeleton);
    if (o.isInstancedMesh) o.dispose();
    for (const m of (Array.isArray(o.material) ? o.material : [o.material])) if (m) mats.add(m); });
  for (const m of mats) { for (const v of Object.values(m)) if (v?.isTexture) textures.add(v); m.dispose(); }
  for (const t of textures) { t.source?.data?.close?.(); t.dispose(); } for (const g of geometries) g.dispose();
  for (const skeleton of skeletons) skeleton.dispose();
}
function restoreMaterials() {
  for (const [mesh, materials] of originalMaterials) mesh.material = materials;
  temporaryMaterials.forEach(m => m.dispose()); temporaryMaterials = [];
}
const channelNames = { beauty: '完整 PBR · 影棚布光', albedo: 'Albedo · 原始颜色', normal: 'Normal · 视空间几何法线', clay: '白模 · 几何与光照' };
const channelNotes = { beauty: '使用模型原始 Albedo、Roughness 与 Metallic 贴图。', albedo: '无光照颜色，保持贴图原始颜色。', normal: '显示视空间几何法线；不是生成的法线贴图。', clay: '临时覆盖为中性漫反射材质，检查形状与布光。' };
function setChannel(value) {
  if (value !== 'beauty') stopPhoto();
  channel = value; restoreMaterials();
  if (value !== 'beauty') for (const [mesh, originals] of originalMaterials) {
    const replace = m => {
      let out;
      if (value === 'albedo') out = new THREE.MeshBasicMaterial({ map: m.map, color: m.color, side: m.side, alphaMap: m.alphaMap, alphaTest: m.alphaTest, transparent: m.transparent, opacity: m.opacity, vertexColors: m.vertexColors, toneMapped: false, fog: false });
      else if (value === 'normal') out = new THREE.MeshNormalMaterial({ side: m.side });
      else out = new THREE.MeshStandardMaterial({ color: '#c6ccc6', metalness: 0, roughness: 0.65, side: m.side });
      temporaryMaterials.push(out); return out;
    };
    mesh.material = Array.isArray(originals) ? originals.map(replace) : replace(originals);
  }
  updateActive('[data-channel]', value, 'channel'); $('#view-label').textContent = channelNames[value]; $('#channel-note').textContent = channelNotes[value];
  $('#render-badge').innerHTML = '<i></i>' + (value === 'beauty' ? 'LIVE PBR' : value.toUpperCase());
  updateFloor();
}

function beginLoad(name) {
  stopPhoto();
  activeFetch?.abort(); activeFetch = null;
  const id = ++requestId; $('#loading').classList.remove('hidden'); $('#error').hidden = true;
  $('#load-title').textContent = '正在读取材质'; $('#load-detail').textContent = name;
  return id;
}
function loadError(error, id) {
  if (id !== requestId || error.name === 'AbortError') return;
  $('#loading').classList.add('hidden'); $('#error').hidden = false;
  $('#error-message').textContent = error.message || '读取失败。请使用包含内嵌贴图的 glTF 2.0 GLB。';
  console.error('Model load failed:', error);
}
function validateGlb(buffer) {
  const fail = () => { throw new Error('这不是有效的 glTF 2.0 GLB 文件。请导出二进制 .glb，并内嵌贴图。'); };
  if (buffer.byteLength < 20) fail();
  const header = new DataView(buffer);
  if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2 || header.getUint32(8, true) !== buffer.byteLength) fail();
  const length = header.getUint32(12, true);
  if (header.getUint32(16, true) !== 0x4e4f534a || length + 20 > buffer.byteLength) fail();
  let json;
  try { json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, length))); } catch (_) { fail(); }
  for (const resource of [...(json.images || []), ...(json.buffers || [])]) {
    // Check before GLTFLoader: image-load failures otherwise become null textures.
    if (resource.uri !== undefined && (typeof resource.uri !== 'string' || !/^data:/i.test(resource.uri))) {
      throw new Error('模型引用了外部贴图或缓冲文件。请重新导出包含内嵌贴图的 GLB。');
    }
  }
}
function validateCandidate(object) {
  let count = 0;
  object.traverse(o => {
    if (o.isLight) o.visible = false; // Studio lighting owns the presentation.
    if (!o.isMesh) return;
    const position = o.geometry?.attributes.position;
    if (!position || !position.count) throw new Error('模型中存在没有有效顶点的网格。');
    for (let i = 0; i < position.count; i++) {
      if (!Number.isFinite(position.getX(i)) || !Number.isFinite(position.getY(i)) || !Number.isFinite(position.getZ(i))) throw new Error('模型顶点包含无效数值。');
    }
    count++;
  });
  if (!count) throw new Error('GLB 中没有可显示的网格。');
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object), size = box.getSize(new THREE.Vector3());
  if (![...box.min, ...box.max].every(Number.isFinite) || !Number.isFinite(size.length()) || size.length() < 1e-9) {
    throw new Error('模型没有可显示的有效几何体。');
  }
  return box;
}
async function loadBuffer(buffer, name, demo, id) {
  if (id !== requestId) return;
  let candidate = null;
  try {
    validateGlb(buffer);
    const gltf = await loader.parseAsync(buffer, '');
    candidate = gltf.scene;
    if (id !== requestId) { disposeAsset(candidate); return; }
    const box = validateCandidate(candidate), size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    const scale = 2.9 / Math.max(size.x, size.y, size.z), candidateMaterials = new Map();
    let candidateTriangles = 0, maxTexture = 0; const materialSet = new Set();
    candidate.traverse(o => { if (!o.isMesh) return;
      // Some GLB exporters omit NORMAL. Reconstruct smooth normals with 60-degree
      // creases instead of showing every simplification triangle in metallic reflections.
      if (!o.geometry.attributes.normal && !o.isSkinnedMesh && !Object.keys(o.geometry.morphAttributes).length) {
        const previousGeometry = o.geometry;
        o.geometry = toCreasedNormals(previousGeometry, Math.PI / 3);
        if (o.geometry !== previousGeometry) previousGeometry.dispose();
        for (const m of (Array.isArray(o.material) ? o.material : [o.material])) { m.flatShading = false; m.needsUpdate = true; }
      }
      o.castShadow = true; o.receiveShadow = true;
      candidateTriangles += (o.geometry.index?.count ?? o.geometry.attributes.position?.count ?? 0) / 3 * (o.isInstancedMesh ? o.count : 1);
      candidateMaterials.set(o, o.material);
      for (const m of (Array.isArray(o.material) ? o.material : [o.material])) { materialSet.add(m);
        for (const prop of ['map', 'metalnessMap', 'roughnessMap', 'normalMap']) {
          const texture = m[prop]; if (!texture) continue;
          texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
          maxTexture = Math.max(maxTexture, texture.image?.width || 0, texture.image?.height || 0);
        }
      }
    });
    // All fallible parsing and bounds checks happen before replacing the visible asset.
    restoreMaterials(); const previous = asset;
    if (previous) orientation.remove(previous);
    asset = candidate; candidate = null; originalMaterials = candidateMaterials; triangles = candidateTriangles;
    orientation.add(asset); orientation.rotation.set(0, 0, 0); orientation.scale.setScalar(scale);
    orientation.position.set(-center.x * scale, -box.min.y * scale + 0.008, -center.z * scale);
    turntable.rotation.set(0, 0, 0); desiredRotation = currentRotation = 0;
    $('#model-angle').value = 0; $('#model-angle-value').textContent = '0°'; updateActive('[data-up]', 'y', 'up');
    orientation.updateMatrixWorld(true); resetCamera();
    if (previous) disposeAsset(previous);
    assetName = demo ? 'Dragonbone Axe' : name.replace(/\.glb$/i, '');
    $('#asset-title').textContent = assetName;
    $('#asset-subtitle').textContent = demo ? '龙骨战斧 · MIKU 生成材质' : '本地 GLB · 原始 PBR 材质';
    $('#triangle-count').textContent = triangles.toLocaleString('en-US', { maximumFractionDigits: 0 });
    $('#texture-size').textContent = maxTexture ? (maxTexture % 1024 === 0 ? `${maxTexture / 1024}K` : `${maxTexture}px`) : '—';
    $('#material-count').textContent = String(materialSet.size).padStart(2, '0');
    $('#demo').classList.toggle('active', demo);
    setChannel('beauty'); $('#loading').classList.add('hidden');
  } catch (error) {
    if (candidate) disposeAsset(candidate);
    loadError(error, id);
  }
}
async function loadUrl(url, name, demo = false) {
  const id = beginLoad(name), controller = new AbortController(); activeFetch = controller;
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error('无法读取 GLB。请检查文件与本地预览服务器。');
    const buffer = await response.arrayBuffer();
    if (id === requestId) await loadBuffer(buffer, name, demo, id);
  } catch (error) { loadError(error, id); }
  finally { if (activeFetch === controller) activeFetch = null; }
}
function loadDemo() { return loadUrl('./assets/axe.glb', 'Dragonbone Axe', true); }
async function openFile(file) {
  if (!file) return;
  if (!/\.glb$/i.test(file.name)) { toast('请选择单个 .glb 文件，纹理需包含在文件内。'); return; }
  if (file.size > 512 * 1024 * 1024) { toast('文件超过 512 MB。建议先简化模型或降低纹理尺寸。'); return; }
  const id = beginLoad(file.name);
  try { await loadBuffer(await file.arrayBuffer(), file.name, false, id); } catch (error) { loadError(error, id); }
}
for (const id of ['open-model', 'drop-button', 'error-retry']) $(`#${id}`).onclick = () => $('#file-input').click();
$('#file-input').onchange = e => { openFile(e.target.files[0]); e.target.value = ''; };
$('#demo').onclick = loadDemo;
$$('[data-preset]').forEach(b => b.onclick = () => setPreset(b.dataset.preset));
$$('[data-channel]').forEach(b => b.onclick = () => setChannel(b.dataset.channel));
$$('[data-up]').forEach(b => b.onclick = () => { if (!asset) return; stopPhoto(); orientation.rotation.set(0, 0, 0); if (b.dataset.up === 'z') orientation.rotation.x = -Math.PI / 2; if (b.dataset.up === 'x') orientation.rotation.z = Math.PI / 2; updateActive('[data-up]', b.dataset.up, 'up'); fitModel(); });
$('#exposure').oninput = e => { renderer.toneMappingExposure = +e.target.value; $('#exposure-value').textContent = (+e.target.value).toFixed(2); };
$('#environment').oninput = e => { scene.environmentIntensity = +e.target.value; $('#environment-value').textContent = (+e.target.value).toFixed(2); syncPhotoLighting(); };
$('#tone-mapping').onchange = e => { renderer.toneMapping = e.target.value === 'neutral' ? THREE.NeutralToneMapping : THREE.AgXToneMapping; };
$('#light-angle').oninput = e => { const a = THREE.MathUtils.degToRad(+e.target.value); scene.environmentRotation.y = a; lightRig.rotation.y = a; $('#light-angle-value').textContent = `${e.target.value}°`; syncPhotoLighting(); };
$('#model-angle').oninput = e => { stopPhoto(); desiredRotation = THREE.MathUtils.degToRad(+e.target.value); $('#model-angle-value').textContent = `${e.target.value}°`; };
$('#reflection').onchange = () => { updateFloor(); syncPhotoLighting(); }; $('#shadows').onchange = updateFloor;
function toggleRotation() { stopPhoto(); controls.autoRotate = !controls.autoRotate; $('#rotate').setAttribute('aria-pressed', String(controls.autoRotate)); }
$('#rotate').onclick = toggleRotation;
$('#reset').onclick = () => { stopPhoto(); desiredRotation = currentRotation = 0; turntable.rotation.y = 0; $('#model-angle').value = 0; $('#model-angle-value').textContent = '0°'; controls.autoRotate = false; $('#rotate').setAttribute('aria-pressed', 'false'); resetCamera(); };
function toggleFocus() { document.body.classList.toggle('focus-mode'); $('#focus').setAttribute('aria-label', document.body.classList.contains('focus-mode') ? '退出沉浸展示' : '沉浸展示'); resize(); }
$('#focus').onclick = toggleFocus;
document.addEventListener('keydown', e => { if (photoExport) { if (e.key === 'Escape') finishPhotoExport(false); return; } if (e.target.matches('input,textarea,select') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key.toLowerCase() === 'f') toggleFocus(); if (e.key.toLowerCase() === 'r') $('#reset').click();
  if (e.code === 'Space') { e.preventDefault(); toggleRotation(); } if (e.key === 'Escape' && document.body.classList.contains('focus-mode')) toggleFocus();
});
let dragDepth = 0;
window.addEventListener('dragenter', e => { if (!e.dataTransfer?.types.includes('Files')) return; e.preventDefault(); dragDepth++; $('#drop-overlay').hidden = false; });
window.addEventListener('dragover', e => { if (!e.dataTransfer?.types.includes('Files')) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
window.addEventListener('dragleave', e => { e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; $('#drop-overlay').hidden = true; } });
window.addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('#drop-overlay').hidden = true; const files = [...e.dataTransfer.files]; if (files.length !== 1) { toast('请一次拖入一个包含贴图的 .glb 文件。'); return; } openFile(files[0]); });

function resize() { if (photoExport) return; const { clientWidth: w, clientHeight: h } = wrap; if (!w || !h) return; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(wrap);
renderer.domElement.addEventListener('webglcontextlost', e => { e.preventDefault(); toast('图形上下文暂时丢失，请刷新页面重新打开模型。'); });
$('#capture').onclick = () => { if (!asset || pendingCapture) return; pendingCapture = true; $('#capture').disabled = true; toast('正在生成 4K 展示图…'); };
function capture() {
  if (photoMode) { startPhotoExport(); return; }
  const oldSize = renderer.getSize(new THREE.Vector2()), oldRatio = renderer.getPixelRatio();
  const maxSide = Math.min(3840, renderer.capabilities.maxTextureSize);
  const w = oldSize.x >= oldSize.y ? maxSide : Math.round(maxSide * oldSize.x / oldSize.y);
  const h = oldSize.x >= oldSize.y ? Math.round(maxSide * oldSize.y / oldSize.x) : maxSide;
  try {
    renderer.setPixelRatio(1); renderer.setSize(w, h, false); renderer.render(scene, camera);
    renderer.domElement.toBlob(blob => {
      if (!blob) { toast('截图生成失败，请重试。'); return; }
      const url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `${assetName.replace(/[^\p{L}\p{N}_-]/gu, '_')}_${activePreset}_${channel}.png`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000); toast(`已导出 ${w} × ${h} PNG`);
    }, 'image/png');
  } catch (e) { toast('截图失败：' + e.message); }
  finally { renderer.setPixelRatio(oldRatio); renderer.setSize(oldSize.x, oldSize.y); pendingCapture = false; $('#capture').disabled = false; }
}

function syncPhotoLighting() {
  if (photoMode && photo) photo.updateLighting(scene, lightRig, presets[activePreset].floor, $('#reflection').checked);
}
function stopPhoto() {
  photoToken++; photoBuilding = false;
  if (photoExport) finishPhotoExport(false);
  photoMode = false; $('#photo-progress').hidden = true;
  $('#render-badge').innerHTML = '<i></i>' + (channel === 'beauty' ? 'LIVE PBR' : channel.toUpperCase()); $('#view-label').textContent = channelNames[channel];
  $('#photo-mode').classList.remove('active'); $('#photo-mode').setAttribute('aria-pressed', 'false'); $('#photo-mode').disabled = false;
  $('#realtime-mode').classList.add('active'); $('#realtime-mode').setAttribute('aria-pressed', 'true');
  $('#render-note').textContent = '实时 HDR 布光 · 拖动查看材质';
  $('#shadows').disabled = false;
}
async function startPhoto() {
  if (!asset || photoBuilding || photoExport) return;
  if (!studioHDR) { toast('摄影模式需要影棚 HDR 环境光，请检查本地资源。'); return; }
  if (triangles > 500000) { toast('摄影模式建议使用 50 万三角形以内的模型，请先简化网格。'); return; }
  const token = ++photoToken; photoBuilding = true; $('#photo-mode').disabled = true;
  $('#photo-progress').hidden = false; $('#photo-status').textContent = '正在准备光线追踪…';
  controls.autoRotate = false; $('#rotate').setAttribute('aria-pressed', 'false'); setChannel('beauty');
  try {
    const { PhotographyRenderer } = await import('./photography.js');
    if (token !== photoToken) return;
    if (!photo) photo = new PhotographyRenderer(renderer, camera);
    turntable.rotation.y = desiredRotation; currentRotation = desiredRotation;
    photo.setScene(turntable, lightRig, studioHDR, scene, presets[activePreset].floor, $('#reflection').checked);
    photoMode = true;
    $('#render-badge').innerHTML = '<i></i>PATH TRACING'; $('#view-label').textContent = '摄影渲染 · 真实间接光与粗糙反射';
    $('#photo-mode').classList.add('active'); $('#photo-mode').setAttribute('aria-pressed', 'true');
    $('#realtime-mode').classList.remove('active'); $('#realtime-mode').setAttribute('aria-pressed', 'false');
    $('#render-note').textContent = '真实间接光与粗糙反射 · 静止后逐步收敛';
    $('#shadows').disabled = true; // Physical shadows are always present in path tracing.
    toast('摄影模式已开启。停下旋转，等待画面逐渐细腻。');
  } catch (error) { stopPhoto(); toast('摄影模式启动失败，已保留实时预览。'); console.error(error); }
  finally { photoBuilding = false; $('#photo-mode').disabled = false; }
}
function startPhotoExport() {
  pendingCapture = false;
  const size = renderer.getSize(new THREE.Vector2()), ratio = renderer.getPixelRatio();
  const longest = Math.min(3840, renderer.capabilities.maxTextureSize);
  const width = Math.round(longest * Math.min(1, size.x / size.y));
  const height = Math.round(longest * Math.min(1, size.y / size.x));
  photoExport = { size, ratio, width, height, renderScale: photo.tracer.renderScale };
  photo.tracer.renderScale = 1;
  document.body.classList.add('photo-exporting'); $('#cancel-photo-export').hidden = false;
  controls.enabled = false; renderer.setPixelRatio(1); renderer.setSize(width, height, false); photo.reset();
}
function finishPhotoExport(save) {
  if (!photoExport) return;
  const old = photoExport;
  if (save) renderer.domElement.toBlob(blob => {
    if (!blob) { toast('截图生成失败，请重试。'); return; }
    const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url;
    a.download = `${assetName.replace(/[^\p{L}\p{N}_-]/gu, '_')}_photographic_4k.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000); toast(`摄影渲染已导出 · ${old.width} × ${old.height} · 128 samples`);
  }, 'image/png');
  photoExport = null; photo.tracer.renderScale = old.renderScale; renderer.setPixelRatio(old.ratio); renderer.setSize(old.size.x, old.size.y);
  controls.enabled = true; $('#capture').disabled = false; $('#cancel-photo-export').hidden = true;
  document.body.classList.remove('photo-exporting'); photo.reset(); resize();
  stopPhoto(); // Release the GPU after an expensive photographic export.
  if (!save) toast('已取消摄影导出。');
}
$('#photo-mode').onclick = startPhoto;
$('#realtime-mode').onclick = stopPhoto;
$('#cancel-photo-export').onclick = () => finishPhotoExport(false);

let lastFrame = 0;
function frame(now) {
  requestAnimationFrame(frame); if (document.hidden) return;
  const delta = Math.min((now - lastFrame) / 1000, 0.1); lastFrame = now;
  if (!photoExport) controls.update(delta); currentRotation = THREE.MathUtils.damp(currentRotation, desiredRotation, 10, delta); turntable.rotation.y = currentRotation;
  if (photoMode && photo) {
    try {
      const limit = 128, samples = photo.render(limit);
      $('#photo-status').textContent = photoExport ? `4K 光追导出 · ${Math.floor(samples)} / ${limit} samples` : photo.tracer.isCompiling ? '正在编译光线追踪…' : `路径追踪 · ${Math.floor(samples)} / ${limit} samples${samples >= limit ? ' · 已收敛' : ' · 静止可提高画质'}`;
      if (photoExport && samples >= limit) finishPhotoExport(true);
    } catch (error) { stopPhoto(); toast('摄影模式暂不可用，已返回实时预览。'); console.error(error); }
  } else renderer.render(scene, camera);
  if (pendingCapture) capture();
}
setPreset('noir'); resetCamera(); resize(); requestAnimationFrame(frame);
try {
  const hdr = await new HDRLoader().loadAsync('./assets/studio_small_09_2k.hdr');
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  const target = pmrem.fromEquirectangular(hdr); environments.set('photo-hdr', target);
  studioEnvironment = target.texture; studioHDR = hdr; setPreset(activePreset);
} catch (error) { console.warn('Studio HDR unavailable; using bundled procedural softboxes.', error); }

const params = new URLSearchParams(location.search);
// The Python helper deliberately exposes a single exact URL, not arbitrary filesystem paths.
if (params.get('model') === '/__model__.glb') {
  loadUrl('/__model__.glb', params.get('name') || 'Exported model.glb');
} else loadDemo();
