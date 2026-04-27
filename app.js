import * as THREE from "https://esm.sh/three@0.177.0";
import { GLTFLoader } from "https://esm.sh/three@0.177.0/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "https://esm.sh/three@0.177.0/examples/jsm/controls/OrbitControls.js";
import { VRMLoaderPlugin } from "https://esm.sh/@pixiv/three-vrm@3.4.2";
import { FaceLandmarker, FilesetResolver } from "https://esm.sh/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL = "./vrms/Iris_z19t_max.vrm";
const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const SMOOTHING = 0.22;
const HEAD_SMOOTHING = 0.16;
const EYE_SMOOTHING = 0.2;
const DIRECT_BLENDSHAPE_THRESHOLD = 0.0005;

const METRIC_KEYS = {
  jawOpen: "jawOpenValue",
  blink: "blinkValue",
  smile: "smileValue",
  head: "headRotationValue",
  state: "faceState",
  binding: "bindingValue",
};

const FALLBACK_EXPRESSION_MAP = {
  eyeBlinkLeft: [["blinkLeft", 1], ["blink", 1]],
  eyeBlinkRight: [["blinkRight", 1], ["blink", 1]],
  eyeSquintLeft: [["blinkLeft", 0.35], ["blink", 0.35]],
  eyeSquintRight: [["blinkRight", 0.35], ["blink", 0.35]],
  browInnerUp: [["surprised", 0.45]],
  browOuterUpLeft: [["surprised", 0.35]],
  browOuterUpRight: [["surprised", 0.35]],
  browDownLeft: [["angry", 0.25]],
  browDownRight: [["angry", 0.25]],
  mouthSmileLeft: [["happy", 0.7], ["relaxed", 0.25]],
  mouthSmileRight: [["happy", 0.7], ["relaxed", 0.25]],
  mouthFrownLeft: [["sad", 0.45]],
  mouthFrownRight: [["sad", 0.45]],
  jawOpen: [["aa", 0.9], ["oh", 0.35], ["ou", 0.2]],
  mouthPucker: [["ou", 0.9]],
  mouthFunnel: [["oh", 0.8]],
  mouthLeft: [["ih", 0.35]],
  mouthRight: [["ih", 0.35]],
};

const state = {
  renderer: null,
  scene: null,
  camera: null,
  clock: new THREE.Clock(),
  controls: null,
  currentVrm: null,
  gazeTarget: null,
  faceLandmarker: null,
  webcamVideo: document.getElementById("webcam"),
  lastVideoTime: -1,
  blendshapeTargetValues: new Map(),
  blendshapeCurrentValues: new Map(),
  expressionTargetValues: new Map(),
  expressionCurrentValues: new Map(),
  morphBindings: new Map(),
  expressionBindings: new Map(),
  lastDetectionTime: 0,
  targetHeadRotation: new THREE.Euler(0, 0, 0, "XYZ"),
  currentHeadRotation: new THREE.Euler(0, 0, 0, "XYZ"),
  headBones: [],
  eyeBones: { left: null, right: null },
  targetEyeRotation: new THREE.Vector2(0, 0),
  currentEyeRotation: new THREE.Vector2(0, 0),
  headWorldPosition: new THREE.Vector3(),
  rigSummary: { morphs: 0, expressions: [] },
};

const ui = {
  startButton: document.getElementById("startButton"),
  statusText: document.getElementById("statusText"),
  jawOpenValue: document.getElementById(METRIC_KEYS.jawOpen),
  blinkValue: document.getElementById(METRIC_KEYS.blink),
  smileValue: document.getElementById(METRIC_KEYS.smile),
  headRotationValue: document.getElementById(METRIC_KEYS.head),
  faceState: document.getElementById(METRIC_KEYS.state),
  bindingValue: document.getElementById(METRIC_KEYS.binding),
};

boot();

function boot() {
  initScene();
  ui.startButton.addEventListener("click", handleStart);
  window.addEventListener("resize", handleResize);
  animate();
}

function initScene() {
  const canvas = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07131f);
  scene.fog = new THREE.Fog(0x07131f, 4.5, 11);

  const camera = new THREE.PerspectiveCamera(30, 4 / 3, 0.1, 100);
  camera.position.set(0, 1.42, 1.8);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.38, 0);
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.minDistance = 1.2;
  controls.maxDistance = 2.5;
  controls.maxPolarAngle = Math.PI * 0.56;
  controls.minPolarAngle = Math.PI * 0.36;

  const hemiLight = new THREE.HemisphereLight(0xa9edff, 0x142231, 1.7);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xfff0d2, 1.9);
  keyLight.position.set(1.2, 1.8, 2.3);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x79dfff, 1.25);
  rimLight.position.set(-1.8, 1.1, -1.4);
  scene.add(rimLight);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(2.3, 48),
    new THREE.MeshStandardMaterial({
      color: 0x0d1f2d,
      transparent: true,
      opacity: 0.9,
      roughness: 0.95,
      metalness: 0.02,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 0);
  scene.add(floor);

  const gazeTarget = new THREE.Object3D();
  gazeTarget.position.set(0, 1.4, 1.0);
  scene.add(gazeTarget);

  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.controls = controls;
  state.gazeTarget = gazeTarget;
  handleResize();
}

async function handleStart() {
  if (ui.startButton.disabled) {
    return;
  }

  ui.startButton.disabled = true;
  setStatus("Initializing", "Requesting camera access and loading the avatar");

  try {
    await Promise.all([setupWebcam(), setupFaceLandmarker(), loadVRM()]);
    setStatus("Tracking", "Live facial expressions and head pose are active");
    ui.faceState.textContent = "Video stream is ready. Move your face into frame.";
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : "Initialization failed";
    setStatus("Startup Failed", message);
    ui.faceState.textContent = "Check camera permission, network access, and the model path.";
    ui.startButton.disabled = false;
  }
}

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 960 },
      height: { ideal: 720 },
    },
  });

  state.webcamVideo.srcObject = stream;
  await state.webcamVideo.play();
}

async function setupFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const sharedOptions = {
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  };

  try {
    state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_LANDMARKER_MODEL,
        delegate: "GPU",
      },
      ...sharedOptions,
    });
  } catch (gpuError) {
    console.warn("GPU delegate unavailable, falling back to CPU.", gpuError);
    state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_LANDMARKER_MODEL,
      },
      ...sharedOptions,
    });
  }
}

async function loadVRM() {
  const loader = new GLTFLoader();
  loader.crossOrigin = "anonymous";
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await loader.loadAsync(MODEL_URL);
  const vrm = gltf.userData.vrm;

  if (!vrm) {
    throw new Error("The loaded file is not a valid VRM.");
  }

  vrm.scene.rotation.y = Math.PI;
  vrm.scene.position.set(0, 0, 0);
  state.scene.add(vrm.scene);
  state.currentVrm = vrm;
  state.headBones = [
    vrm.humanoid?.getNormalizedBoneNode("neck"),
    vrm.humanoid?.getNormalizedBoneNode("head"),
  ].filter(Boolean);
  state.eyeBones = {
    left: vrm.humanoid?.getNormalizedBoneNode("leftEye") ?? null,
    right: vrm.humanoid?.getNormalizedBoneNode("rightEye") ?? null,
  };
  if (vrm.lookAt && state.gazeTarget) {
    vrm.lookAt.target = state.gazeTarget;
  }

  inspectMorphTargets(vrm.scene);
  inspectExpressions(vrm);
  refreshBindingTelemetry();
}

function inspectMorphTargets(root) {
  const bindings = new Map();

  root.traverse((object) => {
    if (!object.isMesh || !object.morphTargetDictionary) {
      return;
    }

    for (const [targetName, index] of Object.entries(object.morphTargetDictionary)) {
      const normalized = normalizeKey(targetName);
      if (!normalized) {
        continue;
      }
      if (!bindings.has(normalized)) {
        bindings.set(normalized, []);
      }
      bindings.get(normalized).push({ mesh: object, index, targetName });
    }
  });

  state.morphBindings = bindings;
  state.rigSummary.morphs = bindings.size;
}

function inspectExpressions(vrm) {
  const manager = vrm.expressionManager;
  const bindings = new Map();

  const mapLike = manager?.expressionMap ?? manager?._expressionMap ?? null;
  if (mapLike) {
    Object.keys(mapLike).forEach((name) => {
      bindings.set(normalizeKey(name), name);
    });
  }

  const expressions = manager?.expressions ?? [];
  expressions.forEach((expression) => {
    if (expression?.expressionName) {
      bindings.set(normalizeKey(expression.expressionName), expression.expressionName);
    }
  });

  state.expressionBindings = bindings;
  state.rigSummary.expressions = Array.from(new Set(bindings.values())).sort();
}

function refreshBindingTelemetry() {
  const expressionNames = state.rigSummary.expressions;
  const preview =
    expressionNames.length > 0
      ? `${expressionNames.slice(0, 6).join(", ")}${expressionNames.length > 6 ? "..." : ""}`
      : "no VRM preset";
  const lookAtMode = state.currentVrm?.lookAt ? "lookAt: enabled" : "lookAt: none";
  ui.bindingValue.textContent = `${state.rigSummary.morphs} morph targets / ${preview} / ${lookAtMode}`;
}

function handleResize() {
  if (!state.renderer || !state.camera) {
    return;
  }

  const canvas = state.renderer.domElement;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 640;
  const height = canvas.clientHeight || Math.round(width * 0.75);

  state.renderer.setSize(width, height, false);
  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);

  const delta = state.clock.getDelta();

  if (state.faceLandmarker && state.webcamVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    runFaceTracking();
  } else {
    fadeTrackingTargets(0.08);
  }

  updateAvatar(delta);
  state.controls?.update();
  state.renderer?.render(state.scene, state.camera);
}

function runFaceTracking() {
  if (state.webcamVideo.currentTime === state.lastVideoTime) {
    return;
  }

  state.lastVideoTime = state.webcamVideo.currentTime;

  const result = state.faceLandmarker.detectForVideo(state.webcamVideo, performance.now());
  const blendshapeCategories = result.faceBlendshapes?.[0]?.categories ?? [];
  const transformMatrix = result.facialTransformationMatrixes?.[0]?.data ?? null;
  const landmarks = result.faceLandmarks?.[0] ?? null;

  if (blendshapeCategories.length === 0) {
    ui.faceState.textContent = "No face detected.";
    fadeTrackingTargets(0.18);
    return;
  }

  state.lastDetectionTime = performance.now();
  ui.faceState.textContent = "Face tracking active.";

  captureBlendshapeTargets(blendshapeCategories);

  if (transformMatrix) {
    captureHeadPose(transformMatrix);
  }

  if (landmarks) {
    captureEyeGaze(landmarks);
  }
}

function captureBlendshapeTargets(categories) {
  const nextDirectTargets = new Map();
  const nextExpressionTargets = new Map();

  categories.forEach(({ categoryName, score }) => {
    const value = clamp(score, 0, 1);
    const normalized = normalizeKey(categoryName);
    nextDirectTargets.set(normalized, value);

    const hasDirectBinding = state.morphBindings.has(normalized);
    const expressionName = state.expressionBindings.get(normalized);

    if (expressionName) {
      nextExpressionTargets.set(expressionName, value);
    }

    if (!hasDirectBinding && !expressionName) {
      const fallbacks = FALLBACK_EXPRESSION_MAP[categoryName] ?? [];
      fallbacks.forEach(([expressionName, multiplier]) => {
        const weighted = clamp(value * multiplier, 0, 1);
        const resolvedExpressionName =
          state.expressionBindings.get(normalizeKey(expressionName)) ?? expressionName;
        const current = nextExpressionTargets.get(resolvedExpressionName) ?? 0;
        nextExpressionTargets.set(resolvedExpressionName, Math.max(current, weighted));
      });
    }
  });

  state.blendshapeTargetValues = nextDirectTargets;
  state.expressionTargetValues = nextExpressionTargets;
  updateTelemetry(categories);
}

function captureHeadPose(matrixData) {
  const matrix = new THREE.Matrix4().fromArray(matrixData);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);

  const rawEuler = new THREE.Euler().setFromQuaternion(quaternion, "XYZ");

  state.targetHeadRotation.x = clamp(-rawEuler.x * 0.75, -0.5, 0.45);
  state.targetHeadRotation.y = clamp(rawEuler.y * 0.9, -0.7, 0.7);
  state.targetHeadRotation.z = clamp(-rawEuler.z * 0.55, -0.35, 0.35);
}

function captureEyeGaze(landmarks) {
  const left = estimateEyeGaze(landmarks, {
    outer: 33,
    inner: 133,
    top: 159,
    bottom: 145,
    iris: [468, 469, 470, 471, 472],
  });
  const right = estimateEyeGaze(landmarks, {
    outer: 263,
    inner: 362,
    top: 386,
    bottom: 374,
    iris: [473, 474, 475, 476, 477],
  });

  if (!left && !right) {
    return;
  }

  const horizontal = clamp((((left?.x ?? 0) + (right?.x ?? 0)) / (left && right ? 2 : 1)) * -0.65, -0.28, 0.28);
  const vertical = clamp((((left?.y ?? 0) + (right?.y ?? 0)) / (left && right ? 2 : 1)) * -0.55, -0.2, 0.2);

  state.targetEyeRotation.set(horizontal, vertical);
}

function estimateEyeGaze(landmarks, indices) {
  const outer = landmarks[indices.outer];
  const inner = landmarks[indices.inner];
  const top = landmarks[indices.top];
  const bottom = landmarks[indices.bottom];
  const irisPoints = indices.iris.map((index) => landmarks[index]).filter(Boolean);

  if (!outer || !inner || !top || !bottom || irisPoints.length === 0) {
    return null;
  }

  const irisCenter = irisPoints.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      return acc;
    },
    { x: 0, y: 0 },
  );
  irisCenter.x /= irisPoints.length;
  irisCenter.y /= irisPoints.length;

  const horizontal = remap(irisCenter.x, outer.x, inner.x);
  const vertical = remap(irisCenter.y, top.y, bottom.y);

  return {
    x: horizontal,
    y: vertical,
  };
}

function updateAvatar(delta) {
  const presenceFade = performance.now() - state.lastDetectionTime > 280 ? 0.12 : 0;
  if (presenceFade > 0) {
    fadeTrackingTargets(presenceFade);
  }

  smoothBlendshapeValues();
  smoothExpressionValues();
  smoothHeadRotation();
  smoothEyeRotation();

  updateGazeTarget();
  state.currentVrm?.update(delta);
  applyBlendshapeWeights();
  applyExpressionWeights();
  applyHeadRotation();
  applyEyeRotation();
}

function smoothBlendshapeValues() {
  const allKeys = new Set([
    ...state.blendshapeCurrentValues.keys(),
    ...state.blendshapeTargetValues.keys(),
    ...state.morphBindings.keys(),
  ]);

  allKeys.forEach((key) => {
    const current = state.blendshapeCurrentValues.get(key) ?? 0;
    const target = state.blendshapeTargetValues.get(key) ?? 0;
    const next = damp(current, target, SMOOTHING);
    state.blendshapeCurrentValues.set(key, next);
  });
}

function smoothExpressionValues() {
  const allKeys = new Set([
    ...state.expressionCurrentValues.keys(),
    ...state.expressionTargetValues.keys(),
    ...state.rigSummary.expressions,
  ]);

  allKeys.forEach((key) => {
    const current = state.expressionCurrentValues.get(key) ?? 0;
    const target = state.expressionTargetValues.get(key) ?? 0;
    const next = damp(current, target, SMOOTHING);
    state.expressionCurrentValues.set(key, next);
  });
}

function smoothHeadRotation() {
  state.currentHeadRotation.x = damp(
    state.currentHeadRotation.x,
    state.targetHeadRotation.x,
    HEAD_SMOOTHING,
  );
  state.currentHeadRotation.y = damp(
    state.currentHeadRotation.y,
    state.targetHeadRotation.y,
    HEAD_SMOOTHING,
  );
  state.currentHeadRotation.z = damp(
    state.currentHeadRotation.z,
    state.targetHeadRotation.z,
    HEAD_SMOOTHING,
  );
}

function smoothEyeRotation() {
  state.currentEyeRotation.x = damp(
    state.currentEyeRotation.x,
    state.targetEyeRotation.x,
    EYE_SMOOTHING,
  );
  state.currentEyeRotation.y = damp(
    state.currentEyeRotation.y,
    state.targetEyeRotation.y,
    EYE_SMOOTHING,
  );
}

function applyBlendshapeWeights() {
  state.blendshapeCurrentValues.forEach((value, normalizedKey) => {
    const bindings = state.morphBindings.get(normalizedKey) ?? [];
    bindings.forEach(({ mesh, index }) => {
      if (!mesh.morphTargetInfluences) {
        return;
      }
      mesh.morphTargetInfluences[index] = value < DIRECT_BLENDSHAPE_THRESHOLD ? 0 : value;
    });
  });
}

function applyExpressionWeights() {
  const manager = state.currentVrm?.expressionManager;
  if (!manager) {
    return;
  }

  state.expressionCurrentValues.forEach((value, expressionName) => {
    try {
      manager.setValue(expressionName, value);
    } catch (error) {
      const mapLike = manager.expressionMap ?? manager._expressionMap;
      if (mapLike?.[expressionName]) {
        mapLike[expressionName].weight = value;
      }
    }
  });
}

function applyHeadRotation() {
  state.headBones.forEach((bone, index) => {
    const influence = index === 0 ? 0.35 : 0.8;
    bone.rotation.x = state.currentHeadRotation.x * influence;
    bone.rotation.y = state.currentHeadRotation.y * influence;
    bone.rotation.z = state.currentHeadRotation.z * influence;
  });

  ui.headRotationValue.textContent = [
    state.currentHeadRotation.x,
    state.currentHeadRotation.y,
    state.currentHeadRotation.z,
  ]
    .map((value) => value.toFixed(2))
    .join(", ");
}

function applyEyeRotation() {
  if (state.currentVrm?.lookAt) {
    return;
  }

  const { left, right } = state.eyeBones;
  if (!left && !right) {
    return;
  }

  const yaw = state.currentEyeRotation.x;
  const pitch = state.currentEyeRotation.y;

  if (left) {
    left.rotation.y = yaw;
    left.rotation.x = pitch;
  }

  if (right) {
    right.rotation.y = yaw;
    right.rotation.x = pitch;
  }
}

function updateGazeTarget() {
  if (!state.gazeTarget) {
    return;
  }

  const headBone = state.headBones[state.headBones.length - 1];
  if (!headBone) {
    return;
  }

  headBone.getWorldPosition(state.headWorldPosition);

  state.gazeTarget.position.set(
    state.headWorldPosition.x + state.currentEyeRotation.x * 0.9,
    state.headWorldPosition.y + state.currentEyeRotation.y * 0.65,
    state.headWorldPosition.z + 1.0,
  );
}

function updateTelemetry(categories) {
  const scores = Object.fromEntries(categories.map(({ categoryName, score }) => [categoryName, score]));
  ui.jawOpenValue.textContent = (scores.jawOpen ?? 0).toFixed(3);
  ui.blinkValue.textContent = `${(scores.eyeBlinkLeft ?? 0).toFixed(3)} / ${(scores.eyeBlinkRight ?? 0).toFixed(3)}`;
  ui.smileValue.textContent = `${(scores.mouthSmileLeft ?? 0).toFixed(3)} / ${(scores.mouthSmileRight ?? 0).toFixed(3)}`;
}

function fadeTrackingTargets(amount) {
  state.blendshapeTargetValues.forEach((value, key) => {
    state.blendshapeTargetValues.set(key, Math.max(0, value - amount));
  });

  state.expressionTargetValues.forEach((value, key) => {
    state.expressionTargetValues.set(key, Math.max(0, value - amount));
  });

  state.targetHeadRotation.x = damp(state.targetHeadRotation.x, 0, amount);
  state.targetHeadRotation.y = damp(state.targetHeadRotation.y, 0, amount);
  state.targetHeadRotation.z = damp(state.targetHeadRotation.z, 0, amount);
  state.targetEyeRotation.x = damp(state.targetEyeRotation.x, 0, amount);
  state.targetEyeRotation.y = damp(state.targetEyeRotation.y, 0, amount);
}

function setStatus(title, detail) {
  ui.statusText.textContent = `${title}｜${detail}`;
}

function normalizeKey(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function damp(current, target, factor) {
  return current + (target - current) * factor;
}

function remap(value, min, max) {
  const span = max - min;
  if (Math.abs(span) < 1e-5) {
    return 0;
  }
  return clamp(((value - min) / span - 0.5) * 2, -1, 1);
}
