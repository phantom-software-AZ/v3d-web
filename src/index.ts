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

import {Engine} from "@babylonjs/core/Engines/engine";

import {FPS} from "@mediapipe/control_utils";
import {Holistic, Results} from "@mediapipe/holistic";

import {Poses} from "./worker/pose-processing";
import {createScene} from "./core";
import {createControlPanel, onResults} from "./mediapipe";
import {test_getBasis, test_quaternionBetweenBases3} from "./helper/utils";


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
const workerPose = Comlink.wrap<Poses>(poseProcessingWorker);
// async function workerInit() {
//     // WebWorkers use `postMessage` and therefore work with Comlink.
//     // workerTestObj = await exposedObj.obj;
//     alert(`Counter: ${workerPose.counter}`);
//     await workerPose.inc(3);
//     await workerPose.spin();
//     alert(`Counter: ${workerPose.counter}`);
// }
//
// workerInit();

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

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading')! as HTMLDivElement;
spinner.ontransitionend = () => {
    spinner.style.display = 'none';
};

window.addEventListener('load', async (e) => {
    console.log("Onload");
    test_quaternionBetweenBases3();
    test_getBasis();

    // v3d
    const vrmManager = await createScene(
        engine,
        workerPose);

    // MediaPipe
    const holistic = new Holistic();
    const mainOnResults = (results: Results) => onResults(
        results,
        vrmManager,
        workerPose,
        videoCanvasElement,
        videoCanvasCtx,
        activeEffect,
        fpsControl
    );
    holistic.initialize().then(() => {
        holistic.onResults(mainOnResults);
    });

    // Present a control panel through which the user can manipulate the solution
    // options.
    createControlPanel(holistic, videoElement, controlsElement, activeEffect, fpsControl);
});


export {};
