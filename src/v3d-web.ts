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
import {Holistic, HolisticConfig, Results} from "@mediapipe/holistic";

import {Poses, poseWrapper} from "./worker/pose-processing";
import {createScene} from "./core";
import {createControlPanel, HolisticOptions, InitHolisticOptions, onResults, setHolisticOptions} from "./mediapipe";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";
import {V3DCore} from "v3d-core/dist/src";


export interface HolisticState {
    initialized: boolean;
    activeEffect: string;
}

export class V3DWeb {
    private readonly worker: Worker;
    private workerPose: Nullable<Comlink.Remote<Poses>> = null;

    private _v3DCore: Nullable<V3DCore> = null;
    get v3DCore(): Nullable<V3DCore>{
        return this._v3DCore;
    }
    private _vrmManager: Nullable<VRMManager> = null;
    get vrmManager(): Nullable<VRMManager>{
        return this._vrmManager;
    }
    private readonly engine: Engine;
    private vrmFilePath: string;

    private readonly fpsControl = this.controlsElement ? new FPS() : null;
    private readonly holistic = new Holistic(this.holisticConfig);
    private holisticState: HolisticState = {
        initialized: false,
        activeEffect: 'mask',
    };
    private _holisticOptions = Object.assign({}, InitHolisticOptions);
    get holisticOptions(): HolisticOptions {
        return this._holisticOptions;
    }
    set holisticOptions(value: HolisticOptions) {
        this._holisticOptions = value;
        setHolisticOptions(value, this.videoElement!, this.holisticState.activeEffect, this.holistic);
    }

    private _noCamera: boolean = false;
    get noCamera(): boolean {
        return this._noCamera;
    }

    constructor(
        vrmFilePath:string,
        public readonly videoElement?: Nullable<HTMLVideoElement>,
        public readonly webglCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly videoCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly controlsElement?: Nullable<HTMLDivElement>,
        private readonly holisticConfig?: HolisticConfig,
    ) {
        let globalInit = false;
        if (!this.videoElement || !this.webglCanvasElement) {
            globalInit = true;
            this.videoElement =
                document.getElementsByClassName('input_video')[0] as HTMLVideoElement;
            this.videoCanvasElement =
                document.getElementById('video-canvas') as HTMLCanvasElement;
            this.webglCanvasElement =
                document.getElementById('webgl-canvas') as HTMLCanvasElement;
            this.controlsElement =
                document.getElementsByClassName('control-panel')[0] as HTMLDivElement;
        }
        if (!this.videoElement || !this.webglCanvasElement) throw Error("Canvas or Video elements not found!");

        this.vrmFilePath = vrmFilePath;

        console.log("v3d-web Onload");
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
         * Comlink/workers
         */
        this.worker = new Worker(
            new URL("./worker/pose-processing", import.meta.url),
            {type: 'module'});
        const posesRemote = Comlink.wrap<typeof poseWrapper>(this.worker);
        const Poses = new posesRemote.poses();
        Poses.then((v) => {
            if (!v) throw Error('Worker start failed!');
            this.workerPose = v;

            createScene(
                this.engine, this.workerPose, this.holistic, this.holisticState,
                this.vrmFilePath, this.videoElement!
            ).then((value) => {
                if (!value) throw Error("VRM Manager initialization failed!");

                const [v3DCore, vrmManager] = value;
                this._v3DCore = v3DCore;
                this._vrmManager = vrmManager;

                // Camera
                this.getVideoDevices().then((devices) => {
                    if (devices.length < 1) {
                        this._noCamera = true;
                    } else {
                        this.getCamera(devices, 0).then(() => {
                            /**
                             * MediaPipe
                             */
                            const mainOnResults = (results: Results) => onResults(
                                results,
                                vrmManager,
                                this.videoCanvasElement,
                                this.workerPose!,
                                this.holisticState.activeEffect,
                                this.fpsControl
                            );
                            this.holistic.initialize().then(() => {
                                // Set initial options
                                setHolisticOptions(this.holisticOptions, this.videoElement!,
                                    this.holisticState.activeEffect, this.holistic);

                                this.holistic.onResults(mainOnResults);
                                this.holisticState.initialized = true;
                            });
                        });
                    }
                });
            });
        });

        // Present a control panel through which the user can manipulate the solution
        // options.
        if (this.controlsElement) {
            createControlPanel(this.holistic, this.videoElement, this.controlsElement,
                this.holisticState.activeEffect, this.fpsControl);
        }
    }

    public async getVideoDevices() {
        // Ask permission
        await navigator.mediaDevices.getUserMedia({video: true});

        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(device => device.kind === 'videoinput');
    }

    public async getCamera(devices: MediaDeviceInfo[], idx: number) {
        // const devices = await this.getVideoDevices();
        await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                deviceId: {
                    exact: devices[idx].deviceId
                }
            }
        })
            .then(stream => {
                this.videoElement!.srcObject = stream;
                this.videoElement!.play();
            });
            // .catch(e => console.error(e));
    }

    /**
     * Close and dispose the application (BabylonJS and MediaPipe)
     */
    public close() {
        this.holisticState.initialized = false;
        this.holistic.close().then(() => {
            this.worker.terminate();
            this.videoElement!.srcObject = null;
        });
        this._vrmManager?.dispose();
        this.engine.dispose();
    }
}

export default V3DWeb;
