/*
Copyright (C) 2021  The v3d Authors.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import * as Comlink from "comlink";

import "@babylonjs/core/Loading/loadingScreen";

// Register plugins (side effect)
import "@babylonjs/core/Loading/Plugins/babylonFileLoader";
import "@babylonjs/core/Materials";
import "@babylonjs/loaders/glTF/glTFFileLoader";

import {Scene} from "@babylonjs/core/scene";
import {Engine} from "@babylonjs/core/Engines/engine";
import {Color3, Vector3} from "@babylonjs/core/Maths/math";

import {
    ControlPanel,
    FPS,
    InputImage,
    Rectangle,
    Slider,
    SourcePicker,
    StaticText,
    Toggle
} from "@mediapipe/control_utils";
import {
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS, FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    HAND_CONNECTIONS, Holistic,
    NormalizedLandmark,
    NormalizedLandmarkList, Options,
    POSE_CONNECTIONS,
    POSE_LANDMARKS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT,
    Results
} from "@mediapipe/holistic";
import {Data, drawConnectors, drawLandmarks, drawRectangle, lerp} from "@mediapipe/drawing_utils";

import {V3DCore} from "v3d-core/dist/src";
import {contain} from "./helper/canvas";
import {exposeObj, obj, poseResults} from "./worker/pose-processing";


/*
 * Global init
 */
const videoElement =
    document.getElementsByClassName('input_video')[0] as HTMLVideoElement;
const webglCanvasElement =
    document.getElementById('webgl-canvas') as HTMLCanvasElement;
const videoCanvasElement =
    document.getElementById('video-canvas') as HTMLCanvasElement;
const controlsElement =
    document.getElementsByClassName('control-panel')[0] as HTMLDivElement;
const videoCanvasCtx = videoCanvasElement.getContext('2d')!;

/*
 * Comlink/workers
 */
const poseProcessingWorker = new Worker(new URL("./worker/pose-processing.ts", import.meta.url),
    {type: 'module'});
const exposedObj = Comlink.wrap<typeof exposeObj>(poseProcessingWorker);
let workerTestObj, workerPose;
async function init() {
    // WebWorkers use `postMessage` and therefore work with Comlink.
    workerTestObj = await exposedObj.obj;
    workerPose = await exposedObj.poseResults;
    alert(`Counter: ${workerTestObj.counter}`);
    workerTestObj.inc(3);
    await workerTestObj.spin();
    alert(`Counter: ${workerTestObj.counter}`);
}

init();

/*
 * Babylonjs
 */
let engine: Engine;
if (Engine.isSupported()) {
    engine = new Engine(webglCanvasElement, true);
}

/*
 * MediaPipe
 */
let activeEffect = 'mask';

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

function removeElements(
    landmarks: NormalizedLandmarkList, elements: number[]) {
    for (const element of elements) {
        delete landmarks[element];
    }
}

function removeLandmarks(results: Results) {
    if (results.poseLandmarks) {
        removeElements(
            results.poseLandmarks,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
    }
}

function connect(
    ctx: CanvasRenderingContext2D,
    connectors:
        Array<[NormalizedLandmark, NormalizedLandmark]>):
    void {
    const canvas = ctx.canvas;
    for (const connector of connectors) {
        const from = connector[0];
        const to = connector[1];
        if (from && to) {
            if (from.visibility && to.visibility &&
                (from.visibility < 0.1 || to.visibility < 0.1)) {
                continue;
            }
            ctx.beginPath();
            ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
            ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
            ctx.stroke();
        }
    }
}

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading')! as HTMLDivElement;
spinner.ontransitionend = () => {
    spinner.style.display = 'none';
};

function onResults(results: Results): void {
    // Hide the spinner.
    document.body.classList.add('loaded');

    // Remove landmarks we don't want to draw.
    removeLandmarks(results);

    // Update the frame rate.
    fpsControl.tick();

    // Draw the overlays.
    videoCanvasCtx.save();
    videoCanvasCtx.clearRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);
    const {
        offsetX,
        offsetY,
        width,
        height
    } = contain(videoCanvasElement.width, videoCanvasElement.height, results.image.width, results.image.height,
        0, 0);

    if (results.segmentationMask) {
        videoCanvasCtx.drawImage(
            results.segmentationMask, 0, 0, videoCanvasElement.width,
            videoCanvasElement.height);

        // Only overwrite existing pixels.
        if (activeEffect === 'mask' || activeEffect === 'both') {
            videoCanvasCtx.globalCompositeOperation = 'source-in';
            // This can be a color or a texture or whatever...
            videoCanvasCtx.fillStyle = '#00FF007F';
            videoCanvasCtx.fillRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);
        } else {
            videoCanvasCtx.globalCompositeOperation = 'source-out';
            videoCanvasCtx.fillStyle = '#0000FF7F';
            videoCanvasCtx.fillRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);
        }

        // Only overwrite missing pixels.
        videoCanvasCtx.globalCompositeOperation = 'destination-atop';
            videoCanvasCtx.drawImage(
            results.image, 0, 0, videoCanvasElement.width, videoCanvasElement.height);

        videoCanvasCtx.globalCompositeOperation = 'source-over';
    } else {
        videoCanvasCtx.drawImage(
            results.image, 0, 0, videoCanvasElement.width, videoCanvasElement.height);
    }

    // Connect elbows to hands. Do this first so that the other graphics will draw
    // on top of these marks.
    videoCanvasCtx.lineWidth = 5;
    if (!!results.poseLandmarks) {
        if (results.rightHandLandmarks) {
            videoCanvasCtx.strokeStyle = 'white';
            connect(videoCanvasCtx, [[
                results.poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW],
                results.rightHandLandmarks[0]
            ]]);
        }
        if (results.leftHandLandmarks) {
            videoCanvasCtx.strokeStyle = 'white';
            connect(videoCanvasCtx, [[
                results.poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW],
                results.leftHandLandmarks[0]
            ]]);
        }

        // Pose...
        drawConnectors(
            videoCanvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(
            videoCanvasCtx,
            Object.values(POSE_LANDMARKS_LEFT)
                .map(index => results.poseLandmarks[index]),
            {visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)'});
        drawLandmarks(
            videoCanvasCtx,
            Object.values(POSE_LANDMARKS_RIGHT)
                .map(index => results.poseLandmarks[index]),
            {visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)'});

        // Hands...
        drawConnectors(
            videoCanvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(videoCanvasCtx, results.rightHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(0,217,231)',
            lineWidth: 2,
            radius: (data: Data) => {
                return lerp(data.from!.z!, -0.15, .1, 10, 1);
            }
        });
        drawConnectors(
            videoCanvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(videoCanvasCtx, results.leftHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(255,138,0)',
            lineWidth: 2,
            radius: (data: Data) => {
                return lerp(data.from!.z!, -0.15, .1, 10, 1);
            }
        });

        // Face...
        // drawConnectors(
        //     videoCanvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
        //     {color: '#C0C0C070', lineWidth: 1});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_IRIS,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYE,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYEBROW,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_IRIS,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYE,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYEBROW,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_FACE_OVAL,
            {color: '#E0E0E0', lineWidth: 5});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LIPS,
            {color: '#E0E0E0', lineWidth: 5});
    }

    videoCanvasCtx.restore();
}

const holistic = new Holistic();
holistic.onResults(onResults);

// Present a control panel through which the user can manipulate the solution
// options.
new ControlPanel(controlsElement, {
    selfieMode: true,
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    effect: 'background',
})
    .add([
        new StaticText({title: 'MediaPipe Holistic'}),
        fpsControl,
        new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
        new SourcePicker({
            onSourceChanged: () => {
                // Resets because the pose gives better results when reset between
                // source changes.
                holistic.reset();
            },
            onFrame:
                async (input: InputImage, size: Rectangle) => {
                    // const aspect = size.height / size.width;
                    // let width: number, height: number;
                    // if (window.innerWidth > window.innerHeight) {
                    //     height = window.innerHeight;
                    //     width = height / aspect;
                    // } else {
                    //     width = window.innerWidth;
                    //     height = width * aspect;
                    // }
                    // videoCanvasElement.width = width;
                    // videoCanvasElement.height = height;
                    await holistic.send({image: input});
                },
        }),
        new Slider({
            title: 'Model Complexity',
            field: 'modelComplexity',
            discrete: ['Lite', 'Full', 'Heavy'],
        }),
        new Toggle(
            {title: 'Smooth Landmarks', field: 'smoothLandmarks'}),
        new Toggle(
            {title: 'Enable Segmentation', field: 'enableSegmentation'}),
        new Toggle(
            {title: 'Smooth Segmentation', field: 'smoothSegmentation'}),
        new Toggle(
            {title: 'Refine Face Landmarks', field: 'refineFaceLandmarks'}),
        new Slider({
            title: 'Min Detection Confidence',
            field: 'minDetectionConfidence',
            range: [0, 1],
            step: 0.01
        }),
        new Slider({
            title: 'Min Tracking Confidence',
            field: 'minTrackingConfidence',
            range: [0, 1],
            step: 0.01
        }),
        new Slider({
            title: 'Effect',
            field: 'effect',
            discrete: {'background': 'Background', 'mask': 'Foreground'},
        }),
    ])
    .on(x => {
        const options = x as Options;
        videoElement.classList.toggle('selfie', options.selfieMode);
        activeEffect = (x as {[key: string]: string})['effect'];
        holistic.setOptions(options);
    });

window.onload = async (e) => {
    console.log("Onload");
    const vrmFile = 'testfiles/2078913627571329107.vrm';

    // Create v3d core
    const v3DCore = new V3DCore(engine, new Scene(engine));
    v3DCore.transparentBackground();
    await v3DCore.AppendAsync('', vrmFile);

    // Get managers
    const vrmManager = v3DCore.getVRMManagerByURI(vrmFile);

    // Camera
    v3DCore.attachCameraTo(vrmManager);

    // Lights
    v3DCore.addAmbientLight(new Color3(1, 1, 1));

    // Lock camera target
    v3DCore.scene.onBeforeRenderObservable.add(() => {
        vrmManager.cameras[0].setTarget(vrmManager.rootMesh.getAbsolutePosition());
    });

    // Render loop
    engine.runRenderLoop(() => {
        v3DCore.scene.render();
    });

    // Model Transformation
    vrmManager.rootMesh.translate(new Vector3(1, 0, 0), 1);
    vrmManager.rootMesh.rotation = new Vector3(0, 135, 0);

    // Work with HumanoidBone
    vrmManager.humanoidBone.leftUpperArm.addRotation(0, -0.5, 0);
    vrmManager.humanoidBone.head.addRotation(0.1, 0, 0);

    // Work with BlendShape(MorphTarget)
    vrmManager.morphing('Joy', 1.0);
};

export {};
