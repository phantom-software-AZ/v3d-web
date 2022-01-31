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

import {Engine, Nullable} from "@babylonjs/core";

import {FPS} from "@mediapipe/control_utils";
import {Holistic, Results} from "@mediapipe/holistic";

import {Poses} from "./worker/pose-processing";
import {createScene} from "./core";
import {createControlPanel, onResults} from "./mediapipe";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";


export interface HolisticState {
    initialized: boolean;
}

export class V3DWeb {
    private readonly worker: Worker;
    private readonly workerPose: Comlink.Remote<Poses>;
    private readonly engine: Engine;

    private activeEffect = 'mask';
    private readonly fpsControl = this.controlsElement ? new FPS() : null;
    private readonly holistic = new Holistic();
    private vrmManager: Nullable<VRMManager> = null;

    private holisticState: HolisticState = {initialized: false};

    constructor(
        public readonly videoElement?: Nullable<HTMLVideoElement>,
        public readonly webglCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly videoCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly controlsElement?: Nullable<HTMLDivElement>,
    ) {
        let globalInit = false;
        if (!this.videoElement || !this.webglCanvasElement) {
            globalInit = true;
            this.videoElement =
                document.getElementsByClassName('input_video')[0] as HTMLVideoElement;
            this.webglCanvasElement =
                document.getElementById('webgl-canvas') as HTMLCanvasElement;
            this.controlsElement =
                document.getElementsByClassName('control-panel')[0] as HTMLDivElement;
        }
        if (!this.videoElement || !this.webglCanvasElement) throw Error("Canvas or Video elements not found!");

        console.log("v3d-web Onload");
        /**
         * Comlink/workers
         */
        this.worker = new Worker(
            new URL("./worker/pose-processing", import.meta.url),
            {type: 'module'});
        this.workerPose = Comlink.wrap<Poses>(this.worker);

        /**
         * Babylonjs
         */
        if (Engine.isSupported()) {
            this.engine = new Engine(this.webglCanvasElement, true);
            // engine.loadingScreen = new CustomLoadingScreen(webglCanvasElement);
        } else {
            throw Error("WebGL is not supported in this browser!");
        }

        // Loading screen
        this.engine.displayLoadingUI();

        /**
         * MediaPipe
         */
        // Present a control panel through which the user can manipulate the solution
        // options.
        if (this.controlsElement) {
            createControlPanel(this.holistic, this.videoElement, this.controlsElement,
                this.activeEffect, this.fpsControl);
        }

        // v3d-core
        createScene(
            this.engine, this.workerPose, this.holistic, this.holisticState).then((v) => {
            this.vrmManager = v;

            const mainOnResults = (results: Results) => onResults(
                results,
                v,
                this.videoCanvasElement,
                this.workerPose,
                this.activeEffect,
                this.fpsControl
            );
            this.holistic.initialize().then(() => {
                this.holistic.onResults(mainOnResults);
                this.holisticState.initialized = true;
            });
        });
    }
}

export default V3DWeb;
