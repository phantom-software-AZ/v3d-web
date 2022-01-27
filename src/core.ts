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
import {ArcRotateCamera, MeshBuilder, Nullable, Quaternion, Scene, StandardMaterial} from "@babylonjs/core";
import {Color3, Vector3} from "@babylonjs/core/Maths";
import {Engine} from "@babylonjs/core/Engines";
import {DebugInfo} from "./helper/debug";

// Debug
import "@babylonjs/core/Debug";
import "@babylonjs/gui";
import "@babylonjs/inspector";
import * as Comlink from "comlink";
import {CloneableQuaternionMapBuffer, Poses} from "./worker/pose-processing";
import {Clock, now} from "./helper/clock";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";
import {HAND_LANDMARKS_BONE_MAPPING} from "./helper/landmark";
import {HumanoidBone} from "v3d-core/dist/src/importer/babylon-vrm-loader/src/humanoid-bone";
import {fixedLengthQueue, KeysMatching} from "./helper/utils";
import {CloneableQuaternionMap, cloneableQuaternionToQuaternion} from "./helper/quaternion";
import {v3DSkyBox} from "v3d-core/dist/src/scene/skybox";
import {InputImage} from "@mediapipe/control_utils";
import {Holistic} from "@mediapipe/holistic";
const IS_DEBUG = true;
export let debugInfo: Nullable<DebugInfo>;
let boneRotations: Nullable<CloneableQuaternionMap> = null,
    holisticUpdate=false,
    bonesNeedUpdate=false,
    textDecode = new TextDecoder();

const videoElement =
    document.getElementsByClassName('input_video')[0] as HTMLVideoElement;

function getVideoDevices() {
    return navigator.mediaDevices.enumerateDevices();
}

async function getCamera() {
    // Ask permission
    await navigator.mediaDevices.getUserMedia({video: true});

    const devices = (await getVideoDevices()).filter(device => device.kind === 'videoinput');
    await navigator.mediaDevices.getUserMedia({
        video: {
            width: 640,
            height: 480,
            deviceId: {
                exact: devices[1].deviceId
            }
        }
    })
        .then(stream => {
            videoElement.srcObject = stream;
            videoElement.play();
        })
        .catch(e => console.error(e));
}

// Can only have one VRM model at this time
export async function createScene(
    engine: Engine,
    workerPose: Comlink.Remote<Poses>,
    holistic: Holistic) {
    await getCamera();

    // const vrmFile = 'testfiles/2078913627571329107.vrm';
    const vrmFile = 'testfiles/Ashtra.vrm';

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
    v3DCore.renderingPipeline.depthOfFieldEnabled = false;

    // Pose web worker
    await workerPose.setBonesHierarchyTree(vrmManager.transformNodeTree);

    // Disable auto animation
    v3DCore.springBonesAutoUpdate = false;
    v3DCore.updateBeforeRenderFunction(
        () => {
            // updateSpringBonesNoForce(vrmManager);
            // Half input fps. This version of Holistic is heavy on CPU time.
            // Wait until they fix web worker (https://github.com/google/mediapipe/issues/2506).
            if (holisticUpdate) {
                holistic.send({image: videoElement})
            }
            holisticUpdate = !holisticUpdate;
        }
    );
    v3DCore.updateAfterRenderFunction(
        () => {
            if (bonesNeedUpdate) {
                updatePose(vrmManager);
                updateSpringBones(vrmManager);
                bonesNeedUpdate = false;
            }
            // updateSpringBonesNoForce(vrmManager);
        }
    );

    // Render loop
    engine.runRenderLoop(() => {
        v3DCore.scene?.render();
    });

    // Model Transformation
    vrmManager.rootMesh.translate(new Vector3(1, 0, 0), 1);
    vrmManager.rootMesh.rotationQuaternion = Quaternion.RotationYawPitchRoll(0, 0, 0);

    // TODO: Debug only
    // @ts-ignore
    window['vrmManager'] = vrmManager;
    // vrmManager.humanoidBone.head.rotationQuaternion = Quaternion.FromEulerAngles(0.5, 0, 0);
    // vrmManager.scene.getTransformNodeByName("Body")?.setEnabled(false);

    // Debug
    if (IS_DEBUG && v3DCore.scene) debugInfo = new DebugInfo(v3DCore.scene);

    return vrmManager;
}

const clock = new Clock();
export function updateSpringBonesNoForce(vrmManager: VRMManager) {
    vrmManager.update(vrmManager.scene.getEngine().getDeltaTime(),
        {dragForce: 1, stiffness: 0, gravityPower: 0});
}
export function updateSpringBones(vrmManager: VRMManager) {
    const deltaTime = clock.getDelta() * 1000;
    // if (deltaTime < 10) return;
    vrmManager.update(deltaTime);
}

export function updateBuffer(data: Uint8Array) {
    const jsonStr = textDecode.decode(data);
    const boneRotationsData: CloneableQuaternionMap = JSON.parse(jsonStr);
    boneRotations = boneRotationsData;
    bonesNeedUpdate = true;
}

export function updatePose(vrmManager: VRMManager) {
    // Wait for buffer to fill
    if (!boneRotations) return;

    const resultBoneRotations = boneRotations;

    vrmManager.morphing('A', boneRotations['mouth'].x);
    vrmManager.morphing('Blink', boneRotations['blink'].z)
    // vrmManager.morphing('Blink_L', await workerPose.blinkLeft);
    // vrmManager.morphing('Blink_R', await workerPose.blinkRight);

    // TODO: option: iris left/right/sync
    if (vrmManager.humanoidBone.leftEye)
        vrmManager.humanoidBone.leftEye.rotationQuaternion = cloneableQuaternionToQuaternion(
            boneRotations['iris']);
    if (vrmManager.humanoidBone.rightEye)
        vrmManager.humanoidBone.rightEye.rotationQuaternion = cloneableQuaternionToQuaternion(
            boneRotations['iris']);

    const left = 'left';
    for (const [k, v] of Object.entries(HAND_LANDMARKS_BONE_MAPPING)) {
        const key = left + k as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        if (vrmManager.humanoidBone[key]) {
            vrmManager.humanoidBone[key]!.rotationQuaternion = cloneableQuaternionToQuaternion(
                resultBoneRotations[key]);
        }
    }

    const right = 'right';
    for (const [k, v] of Object.entries(HAND_LANDMARKS_BONE_MAPPING)) {
        const key = right + k as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        if (vrmManager.humanoidBone[key]) {
            vrmManager.humanoidBone[key]!.rotationQuaternion = cloneableQuaternionToQuaternion(
                resultBoneRotations[key]);
        }
    }

    vrmManager.humanoidBone.hips.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['hips']);
    vrmManager.humanoidBone.spine.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['spine']);
    vrmManager.humanoidBone.neck.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['neck']);
    vrmManager.humanoidBone.head.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['head']);

    const lr = ["left", "right"];
    // Arms
    for (const k of lr) {
        const upperArmKey = `${k}UpperArm` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        const lowerArmKey = `${k}LowerArm` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[upperArmKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[upperArmKey]);
        vrmManager.humanoidBone[lowerArmKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[lowerArmKey]);
        // Legs
        const upperLegKey = `${k}UpperLeg` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        const lowerLegKey = `${k}LowerLeg` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[upperLegKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[upperLegKey]);
        vrmManager.humanoidBone[lowerLegKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[lowerLegKey]);
        // Feet
        const footKey = `${k}Foot` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[footKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[footKey]);
    }
}
