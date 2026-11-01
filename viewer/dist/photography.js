import * as THREE from 'three';
import { WebGLPathTracer } from './vendor/pathtracer.module.js';

function studioSweep() {
  const profile = [[60, 0], [-5, 0]];
  for (let i = 1; i <= 40; i++) {
    const a = i / 40 * Math.PI / 2;
    profile.push([-5 - 4 * Math.sin(a), 4 * (1 - Math.cos(a))]);
  }
  profile.push([-9, 60]);
  const positions = [], indices = [];
  for (const [z, y] of profile) positions.push(-60, y, z, 60, y, z);
  for (let i = 0; i < profile.length - 1; i++) {
    const a = i * 2; indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals(); return geometry;
}

// Keep the physical photographic scene separate from the real-time reflector
// and shadow catcher. The original GLB's material textures are shared unchanged.
export class PhotographyRenderer {
  constructor(renderer, camera) {
    this.renderer = renderer; this.camera = camera;
    this.tracer = new WebGLPathTracer(renderer);
    this.tracer.bounces = 6;
    this.tracer.transmissiveBounces = 8;
    this.tracer.filterGlossyFactor = 0.35;
    this.tracer.textureSize.set(2048, 2048);
    this.tracer.tiles.set(3, 3);
    this.tracer.renderDelay = 150;
    this.tracer.minSamples = 2;
    this.tracer.fadeDuration = 250;
    const bufferSize = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.tracer.renderScale = Math.min(1, 1600 / Math.max(bufferSize.x, bufferSize.y));
    this.tracer.dynamicLowRes = true;
    this.tracer.lowResScale = 0.3;
    this.cameraMatrix = new THREE.Matrix4();
    this.projectionMatrix = new THREE.Matrix4();
    this.scene = new THREE.Scene();
    this.floor = new THREE.Mesh(studioSweep(), new THREE.MeshPhysicalMaterial({ color: '#171d20', roughness: 0.3, metalness: 0, ior: 1.4, specularIntensity: 0.16, side: THREE.DoubleSide }));
    this.floor.position.y = -0.012;
    this.model = null; this.lights = null;
  }
  setScene(model, lights, hdr, sourceScene, floorColor, reflective) {
    if (this.model) this.scene.remove(this.model);
    if (this.lights) this.scene.remove(this.lights);
    this.model = model.clone(true); this.lights = lights.clone(true);
    this.model.traverse(o => { if (o.isLight) o.visible = false; });
    this.scene.add(this.model, this.lights, this.floor);
    this.scene.background = sourceScene.background.clone();
    this.scene.environment = hdr;
    this.scene.environmentIntensity = sourceScene.environmentIntensity;
    this.scene.environmentRotation.copy(sourceScene.environmentRotation);
    this.floor.material.color.set(floorColor);
    this.floor.material.roughness = reflective ? 0.3 : 1;
    this.floor.material.specularIntensity = reflective ? 0.16 : 0;
    this.tracer.setScene(this.scene, this.camera);
    this.cameraMatrix.copy(this.camera.matrixWorld); this.projectionMatrix.copy(this.camera.projectionMatrix);
    this.tracer.pausePathTracing = false;
  }
  updateLighting(sourceScene, lights, floorColor, reflective) {
    this.scene.background.copy(sourceScene.background);
    this.scene.environmentIntensity = sourceScene.environmentIntensity;
    this.scene.environmentRotation.copy(sourceScene.environmentRotation);
    if (this.lights) this.scene.remove(this.lights);
    this.lights = lights.clone(true); this.scene.add(this.lights); this.scene.updateMatrixWorld(true);
    this.floor.material.color.set(floorColor);
    this.floor.material.roughness = reflective ? 0.3 : 1;
    this.floor.material.specularIntensity = reflective ? 0.16 : 0;
    this.tracer.updateEnvironment(); this.tracer.updateLights(); this.tracer.updateMaterials();
    this.tracer.pausePathTracing = false;
  }
  render(limit = 128) {
    this.camera.updateMatrixWorld();
    if (this.camera.matrixWorld.elements.some((v, i) => Math.abs(v - this.cameraMatrix.elements[i]) > 1e-5) || this.camera.projectionMatrix.elements.some((v, i) => Math.abs(v - this.projectionMatrix.elements[i]) > 1e-5)) {
      this.cameraMatrix.copy(this.camera.matrixWorld); this.projectionMatrix.copy(this.camera.projectionMatrix);
      this.tracer.updateCamera();
    }
    this.tracer.pausePathTracing = this.tracer.samples >= limit;
    this.tracer.renderSample();
    return this.tracer.samples;
  }
  reset() { this.tracer.reset(); this.tracer.pausePathTracing = false; }
  dispose() { this.tracer.dispose(); this.floor.geometry.dispose(); this.floor.material.dispose(); }
}
