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

import {Engine, Nullable, ArcRotateCamera, Vector3, Quaternion} from "@babylonjs/core";

import {FPS} from "@mediapipe/control_utils";
import {Holistic, HolisticConfig, Results} from "@mediapipe/holistic";

import {Poses, poseWrapper} from "./worker/pose-processing";
import {createScene, updateBuffer, updatePose, updateSpringBones} from "./core";
import {createControlPanel, HolisticOptions, InitHolisticOptions, onResults, setHolisticOptions} from "./mediapipe";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";
import {V3DCore} from "v3d-core/dist/src";
import {CloneableQuaternionMap} from "./helper/quaternion";


export interface HolisticState {
    ready: boolean;
    activeEffect: string;
    holisticUpdate: boolean;
}
export interface BoneState {
    boneRotations: Nullable<CloneableQuaternionMap>;
    bonesNeedUpdate: boolean;
}
export interface BoneOptions {
    irisLinkLR: boolean;
    irisLockX: boolean;
    lockFinger: boolean;
    lockArm: boolean;
    lockLeg: boolean;
    resetInvisible: boolean;
}

export class V3DWeb {
    private readonly worker: Worker;
    private workerPose: Nullable<Comlink.Remote<Poses>> = null;
    private boneState: BoneState = {
        boneRotations: null,
        bonesNeedUpdate: false,
    }
    private _boneOptions: BoneOptions = {
        irisLinkLR: true,
        irisLockX: true,
        lockFinger: false,
        lockArm: false,
        lockLeg: false,
        resetInvisible: false,
    }
    get boneOptions(): BoneOptions {
        return this._boneOptions;
    }
    set boneOptions(value: BoneOptions) {
        this._boneOptions = value;
    }
    private readonly _updateBufferCallback = Comlink.proxy((data: Uint8Array) => {
        updateBuffer(data, this.boneState)
    });

    private _v3DCore: Nullable<V3DCore> = null;
    get v3DCore(): Nullable<V3DCore>{
        return this._v3DCore;
    }
    private _vrmManager: Nullable<VRMManager> = null;
    get vrmManager(): Nullable<VRMManager>{
        return this._vrmManager;
    }
    private readonly engine: Engine;
    private _vrmFile: File | string;
    set vrmFile(value: File | string) {
        this._vrmFile = value;
        this.switchModel();
    }

    private readonly fpsControl = this.controlsElement ? new FPS() : null;
    private readonly holistic = new Holistic(this.holisticConfig);
    private holisticState: HolisticState = {
        ready: false,
        activeEffect: 'mask',
        holisticUpdate: false,
    };
    private _holisticOptions = Object.assign({}, InitHolisticOptions);
    get holisticOptions(): HolisticOptions {
        return this._holisticOptions;
    }
    set holisticOptions(value: HolisticOptions) {
        this._holisticOptions = value;
        setHolisticOptions(value, this.videoElement!, this.holisticState.activeEffect, this.holistic);
    }

    private _cameraList: MediaDeviceInfo[] = [];
    get cameraList(): MediaDeviceInfo[] {
        return this._cameraList;
    }

    constructor(
        vrmFilePath:string,
        public readonly videoElement?: Nullable<HTMLVideoElement>,
        public readonly webglCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly videoCanvasElement?: Nullable<HTMLCanvasElement>,
        public readonly controlsElement?: Nullable<HTMLDivElement>,
        private readonly holisticConfig?: HolisticConfig,
        afterInitCallback?: (...args : any[]) => any,
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

        this._vrmFile = vrmFilePath;

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
        const Poses = new posesRemote.poses(this._updateBufferCallback);
        Poses.then((v) => {
            if (!v) throw Error('Worker start failed!');
            this.workerPose = v;

            createScene(
                this.engine, this.workerPose, this.boneState,
                this.holistic, this.holisticState,
                this._vrmFile, this.videoElement!
            ).then((value) => {
                if (!value) throw Error("VRM Manager initialization failed!");

                const [v3DCore, vrmManager] = value;
                this._v3DCore = v3DCore;
                this._vrmManager = vrmManager;

                // Camera
                this.getVideoDevices().then((devices) => {
                    if (devices.length < 1) {
                        throw Error("No camera found!");
                    } else {
                        this._cameraList = devices;
                        this.getCamera(0).then(() => {
                            /**
                             * MediaPipe
                             */
                            const mainOnResults = (results: Results) => onResults(
                                results,
                                vrmManager,
                                this.videoCanvasElement,
                                this.workerPose!,
                                this.holisticState.activeEffect,
                                this._updateBufferCallback,
                                this.fpsControl
                            );
                            this.holistic.initialize().then(() => {
                                // Set initial options
                                setHolisticOptions(this.holisticOptions, this.videoElement!,
                                    this.holisticState.activeEffect, this.holistic);

                                this.holistic.onResults(mainOnResults);
                                this.holisticState.ready = true;

                                if (afterInitCallback) afterInitCallback();
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

    public async getCamera(idx: number) {
        await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                deviceId: {
                    exact: this.cameraList[idx].deviceId
                }
            }
        })
            .then(stream => {
                if (!this.videoElement) throw Error("Video Element not found!");
                this.videoElement.srcObject = stream;
                // let source = document.createElement('source');
                //
                // source.setAttribute('src', 'testfiles/dance5.webm');
                // source.setAttribute('type', 'video/webm');
                //
                // this.videoElement.appendChild(source);
                this.videoElement.play();
            });
            // .catch(e => console.error(e));
    }

    /**
     * Close and dispose the application (BabylonJS and MediaPipe)
     */
    public close() {
        this.holisticState.ready = false;
        this.holistic.close().then(() => {
            this.worker.terminate();
            (this._updateBufferCallback as any) = null;
            this.videoElement!.srcObject = null;
        });
        this._vrmManager?.dispose();
        this.engine.dispose();
    }

    /**
     * Reset poses and holistic
     */
    public reset() {
        this.workerPose?.resetBoneRotations(true);
        this.holistic.reset();
    }

    public switchSource(idx: number) {
        if (idx >= this.cameraList.length) return;

        this.holisticState.ready = false;
        this.getCamera(idx).then(() => {
            this.reset();
            this.holisticState.ready = true;
        });
    }

    private async switchModel() {
        if (!this.v3DCore || !this.workerPose) return;
        this.v3DCore.updateAfterRenderFunction(() => {});
        this.vrmManager?.dispose();
        this._vrmManager = null;

        await this.v3DCore.AppendAsync('', this._vrmFile);
        this._vrmManager = this.v3DCore.getVRMManagerByURI((this._vrmFile as File).name ?
            (this._vrmFile as File).name : (this._vrmFile as string));

        if (!this._vrmManager) throw Error("VRM model loading failed!");

        // Reset camera
        const mainCamera = (this.v3DCore.mainCamera as ArcRotateCamera);
        mainCamera.setPosition(new Vector3(0, 1.05, 4.5));
        mainCamera.setTarget(
            this._vrmManager.rootMesh.getWorldMatrix().getTranslation().subtractFromFloats(0, -1.25, 0));
        await this.workerPose.setBonesHierarchyTree(this._vrmManager.transformNodeTree, true);
        this.workerPose.resetBoneRotations();
        this.v3DCore.updateAfterRenderFunction(
            () => {
                if (this.boneState.bonesNeedUpdate) {
                    updatePose(this._vrmManager!, this.boneState);
                    updateSpringBones(this._vrmManager!);
                    this.boneState.bonesNeedUpdate = false;
                }
            }
        );
        this._vrmManager.rootMesh.rotationQuaternion = Quaternion.RotationYawPitchRoll(0, 0, 0);
    }
}

export default V3DWeb;
