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

import {V3DCore} from "v3d-core/dist/src";
import {ArcRotateCamera, Mesh, Nullable, Scene} from "@babylonjs/core";
import {Color3, Vector3} from "@babylonjs/core/Maths";
import {Engine} from "@babylonjs/core/Engines";
import {makeSphere} from "./helper/debug";
import {Arrow3D, initArray, normalizedLandmarkToVector, POSE_LANDMARK_LENGTH} from "./helper/utils";

// Debug
import "@babylonjs/core/Debug";
import "@babylonjs/gui";
import "@babylonjs/inspector";
import {NormalizedLandmarkList, POSE_LANDMARKS} from "@mediapipe/holistic";
const IS_DEBUG = true;
export let debugInfo: Nullable<DebugInfo>;

export async function createScene(engine: Engine) {
    const vrmFile = 'testfiles/2078913627571329107.vrm';

    // Create v3d core
    const v3DCore = new V3DCore(engine, new Scene(engine));
    v3DCore.transparentBackground();
    await v3DCore.AppendAsync('', vrmFile);

    // Get managers
    const vrmManager = v3DCore.getVRMManagerByURI(vrmFile);

    // Camera
    v3DCore.attachCameraTo(vrmManager);
    (v3DCore.mainCamera as ArcRotateCamera).setPosition(new Vector3(0, 0, -5));
    (v3DCore.mainCamera as ArcRotateCamera).setTarget(Vector3.Zero());

    // Lights
    v3DCore.addAmbientLight(new Color3(1, 1, 1));

    // Lock camera target
    v3DCore.scene?.onBeforeRenderObservable.add(() => {
        vrmManager.cameras[0].setTarget(vrmManager.rootMesh.getAbsolutePosition());
    });

    // Render loop
    engine.runRenderLoop(() => {
        v3DCore.scene?.render();
    });

    // Model Transformation
    vrmManager.rootMesh.translate(new Vector3(1, 0, 0), 1);
    vrmManager.rootMesh.rotation = new Vector3(0, 135, 0);

    // Work with HumanoidBone
    vrmManager.humanoidBone.leftUpperArm.addRotation(0, -0.5, 0);
    vrmManager.humanoidBone.head.addRotation(0.1, 0, 0);

    // Work with BlendShape(MorphTarget)
    vrmManager.morphing('Joy', 1.0);

    // Debug
    if (IS_DEBUG && v3DCore.scene) debugInfo = new DebugInfo(v3DCore.scene);
}

class DebugInfo {
    public poseLandmarkSpheres: Mesh[];
    public faceNormalArrows: Arrow3D[];

    constructor(
        private readonly scene: Scene
    ) {
        this.poseLandmarkSpheres = this.initPoseLandmarks();
        this.faceNormalArrows = this.initFaceNormalArrows();
        scene.debugLayer.show({
            globalRoot: document.getElementById('wrapper') as HTMLElement,
            handleResize: true,
        });
    }

    private initPoseLandmarks() {
        return initArray<Mesh>(
            POSE_LANDMARK_LENGTH,
            () => makeSphere(
                this.scene, Vector3.One(), undefined, {diameter: 0.05}));
    }

    private initFaceNormalArrows() {
        return initArray<Arrow3D>(
            1,    // Temp magical number
            () => new Arrow3D(this.scene,
                0.02, 32, 0.08, 0.08,
                0.5, Vector3.Zero(), Vector3.One()));
    }

    public updatePoseLandmarkSpheres(resultPoseLandmarks: NormalizedLandmarkList) {
        if (resultPoseLandmarks.length != POSE_LANDMARK_LENGTH) return;
        for (let i = 0; i < POSE_LANDMARK_LENGTH; ++i) {
            this.poseLandmarkSpheres[i].position.set(
                resultPoseLandmarks[i].x,
                resultPoseLandmarks[i].y,
                resultPoseLandmarks[i].z
            );
        }
    }

    public updateFaceNormalArrows(
        resultFaceNormals: NormalizedLandmarkList,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        if (resultFaceNormals.length != this.faceNormalArrows.length) return;
        for (let i = 0; i < this.faceNormalArrows.length; ++i) {
            this.faceNormalArrows[i].updateStartAndDirection(
                normalizedLandmarkToVector(
                    resultPoseLandmarks[POSE_LANDMARKS.NOSE]),
                normalizedLandmarkToVector(resultFaceNormals[i]),
            );
        }
    }
}
