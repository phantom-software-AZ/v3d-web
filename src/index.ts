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

import {Engine} from "@babylonjs/core";

import {FPS} from "@mediapipe/control_utils";
import {Holistic, Results} from "@mediapipe/holistic";

import {Poses} from "./worker/pose-processing";
import {createScene} from "./core";
import {createControlPanel, onResults} from "./mediapipe";
import {CustomLoadingScreen} from "./helper/utils";


/*
 * Global init
 */
const videoElement =
    document.getElementsByClassName('input_video')[0] as HTMLVideoElement;
const webglCanvasElement =
    document.getElementById('webgl-canvas') as HTMLCanvasElement;
const controlsElement =
    document.getElementsByClassName('control-panel')[0] as HTMLDivElement;

/*
 * Comlink/workers
 */
const poseProcessingWorker = new Worker(new URL("./worker/pose-processing.ts", import.meta.url),
    {type: 'module'});
const workerPose = Comlink.wrap<Poses>(poseProcessingWorker);

/*
 * Babylonjs
 */
let engine: Engine;
if (Engine.isSupported()) {
    engine = new Engine(webglCanvasElement, true);
    engine.loadingScreen = new CustomLoadingScreen(webglCanvasElement);
}

/*
 * MediaPipe
 */
let activeEffect = 'mask';

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

window.addEventListener('load', async () => {
    console.log("Onload");

    // Loading screen
    engine.displayLoadingUI();

    // MediaPipe
    const holistic = new Holistic();
    const mainOnResults = (results: Results) => onResults(
        results,
        vrmManager,
        workerPose,
        activeEffect,
        fpsControl
    );
    holistic.initialize().then(() => {
        holistic.onResults(mainOnResults);
    });

    // Present a control panel through which the user can manipulate the solution
    // options.
    createControlPanel(holistic, videoElement, controlsElement, activeEffect, fpsControl);

    // v3d
    const vrmManager = await createScene(
        engine, workerPose, holistic);
});


export {};
